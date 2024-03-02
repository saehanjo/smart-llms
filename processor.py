from functools import partial
from enum import Enum
import itertools
import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom, norm
import time

import utils_math


class Processor:
    methods = ["naive", "exhaustive", "efficient", "multi-model"]
    best_model = "gpt-4"

    def __init__(self, method, model2info):
        self.method = method
        self.model2info = model2info
        # Sort the models based on the cost.
        self.ordered_models = sorted(
            model2info, key=lambda k: model2info[k].prompt_cost
        )
        self.time_opt_ci = 0
        self.time_opt_expected = 0
        self.time_opt_program = 0

    def process(self, df, acc_diff, confidence):
        assert acc_diff >= 0 and acc_diff <= 1
        assert confidence >= 0 and confidence <= 1

        if self.method == "naive":
            return self.naive(df)
        elif self.method == "exhaustive":
            return self.exhaustive(df, acc_diff, confidence)
        elif self.method == "efficient":
            return self.efficient(df, acc_diff, confidence)
        elif self.method == "multi-model":
            return self.multi_model(df, acc_diff, confidence)
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def naive(self, df, model=None):
        model = Processor.best_model if model is None else model

        result = ModelResult(model)
        result.status = ModelStatus.satisfied
        cost = 0
        for idx, row in df.iterrows():
            # nr_prompt_tokens = self.model2info[model].get_nr_tokens(row['text'])
            nr_prompt_tokens = row["nr_prompt_tokens"][model]
            nr_completion_tokens = row["nr_completion_tokens"][model]
            result.add_instance(idx, row[model], nr_prompt_tokens, nr_completion_tokens)
            cost += self.model2info[model].calculate_cost(
                nr_prompt_tokens, nr_completion_tokens
            )
        return cost, {model: result}

    def get_profile_models(self, model2result):
        # Select unknown models that are cheaper than the smallest satisfied model.
        idx_smallest_model_satisfied = [
            idx
            for idx, model in enumerate(self.ordered_models)
            if model2result[model].status == ModelStatus.satisfied
        ][0]
        unknown_models = [
            model
            for idx, model in enumerate(self.ordered_models)
            if model2result[model].status == ModelStatus.unknown
            and idx <= idx_smallest_model_satisfied
        ]
        return unknown_models

    def find_smallest_model(self, df, acc_diff, confidence, use_expected=False):
        best_model = Processor.best_model

        cost = 0
        smallest_model = None
        cnt = None
        model2result = {model: ModelResult(model) for model in self.ordered_models}
        model2result[best_model].status = ModelStatus.satisfied
        model2conform = {
            model: 0 for model in self.ordered_models if model != best_model
        }
        # Find smallest model that satisfies the accuracy requirement.
        for idx, row in df.iterrows():
            # Select models for profiling.
            valid_models = self.get_profile_models(model2result) + [best_model]
            # Get output per model.
            for model in valid_models:
                # nr_prompt_tokens = self.model2info[model].get_nr_tokens(row['text'])
                nr_prompt_tokens = row["nr_prompt_tokens"][model]
                nr_completion_tokens = row["nr_completion_tokens"][model]
                model2result[model].add_instance(
                    idx, row[model], nr_prompt_tokens, nr_completion_tokens
                )
                cost += self.model2info[model].calculate_cost(
                    nr_prompt_tokens, nr_completion_tokens
                )
            # Add to the number of outputs that match those of the best model.
            out_best_model = model2result[best_model].idx2out[idx]
            for model in valid_models:
                if model != best_model:
                    if model2result[model].idx2out[idx] == out_best_model:
                        model2conform[model] += 1
            # Check if model conforms with the best model.
            if (idx + 1) % 10 == 0: # or idx == 0:
                start_time = time.time()
                model2ci = {
                    model: utils_math.binom_ci(nr_conform, idx + 1, confidence)
                    for model, nr_conform in model2conform.items()
                    if model in valid_models and model != best_model
                }
                self.time_opt_ci += time.time() - start_time
                for model, (l, u) in model2ci.items():
                    if (
                        (idx + 1) % 10 == 0 or u < 1 - acc_diff or l >= 1 - acc_diff # or idx == 0
                    ) and model2result[model].status == ModelStatus.unknown:
                        print(
                            f"Idx: {idx}, Model: {model}, Same: {model2conform[model]}, Low: {l}, High: {u}"
                        )
                    if u < 1 - acc_diff:
                        model2result[model].status = ModelStatus.invalid
                        print(f"Removed {model} from valid models.")
                    elif (
                        l >= 1 - acc_diff
                        and model2result[model].status == ModelStatus.unknown
                    ):
                        model2result[model].status = ModelStatus.satisfied
                        print(f"Able to use {model} for all examples.")

                if use_expected and (idx + 1) >= 100:
                    # Prune models based on expected cost per model.
                    self.prune_models(
                        idx + 1,
                        len(df),
                        model2result,
                        model2conform,
                        best_model,
                        acc_diff,
                        confidence,
                    )
                    # self.prune_models_prev(
                    #     idx + 1,
                    #     len(df),
                    #     model2result,
                    #     model2conform,
                    #     best_model,
                    #     acc_diff,
                    #     confidence,
                    # )

                # Check if we found the smallest model.
                for model in self.ordered_models:
                    if model2result[model].status == ModelStatus.invalid:
                        continue
                    elif model2result[model].status == ModelStatus.unknown:
                        break
                    elif model2result[model].status == ModelStatus.satisfied:
                        smallest_model = model
                        break
                if smallest_model:
                    cnt = idx + 1
                    break

        # Find the smallest model if not determined yet.
        if smallest_model is None:
            for model in self.ordered_models:
                if model2result[model].status == ModelStatus.satisfied:
                    smallest_model = model
                    cnt = len(df)
                    break
        return smallest_model, cost, cnt, model2result

    def expected_cost(
        self,
        k,
        unknown_models,
        model_satisfied,
        cnt,
        nr_items_total,
        model2conform,
        best_model,
        acc_diff,
        confidence,
        model2unitcost,
    ):
        # Compute binomial threshold per model.
        model2bt = {
            # model: utils_math.binom_threshold_gaussian(
            #     model2conform[model],
            #     cnt + k,
            #     1 - acc_diff,
            #     norm.ppf(1 - (1 - confidence) / 2),
            # )
            model: utils_math.binom_threshold(
                model2conform[model], cnt + k, acc_diff, confidence
            )
            for model in unknown_models
        }
        # Compute probability of satisfying the accuracy constraint.
        model2acc = {model: model2conform[model] / cnt for model in unknown_models}
        # # Use single binomial distribution.
        # model2prob = {
        #     model: binom.sf(model2bt[model] - 1, k, model2acc[model])
        #     for model in unknown_models
        # }
        # Use integral.
        model2std = {
            model: (p * (1 - p) / cnt) ** 0.5 for model, p in model2acc.items()
        }
        model2prob = {
            model: norm.expect(
                partial(binom.sf, model2bt[model] - 1, k),
                loc=model2acc[model],
                scale=model2std[model],
                lb=0.0,
                ub=1.0,
            )
            for model in unknown_models
        }
        # print(f"Current accuracy: {model2conform}, cnt: {cnt}, k: {k}")
        # print(f"Binomial thresholds: {model2bt}, k: {k}")
        # print(f"Probabilities: {model2prob}, k: {k}")
        # Compute expected cost of profiling k items over all models.
        profile_cost = k * (
            model2unitcost[best_model]
            + sum([model2unitcost[model] for model in unknown_models])
        )
        # These unknown models are ordered by prompt cost.
        apply_cost = 0
        prob_complement = 1
        for model in unknown_models:
            apply_cost += (
                prob_complement
                * model2prob[model]
                * (nr_items_total - cnt - k)
                * model2unitcost[model]
            )
            prob_complement *= 1 - model2prob[model]
        apply_cost += (
            prob_complement
            * (nr_items_total - cnt - k)
            * model2unitcost[model_satisfied]
        )
        expected = profile_cost + apply_cost
        return expected, profile_cost, apply_cost

    def min_expected_cost(
        self,
        unknown_models,
        model_satisfied,
        cnt,
        nr_items_total,
        model2conform,
        best_model,
        acc_diff,
        confidence,
        model2unitcost,
    ):
        min_expected = float("inf")
        k_expected = None
        max_k = nr_items_total - cnt
        k = 1
        while k < max_k:
            start_time = time.time()
            expected, profile_cost, apply_cost = self.expected_cost(
                k,
                unknown_models,
                model_satisfied,
                cnt,
                nr_items_total,
                model2conform,
                best_model,
                acc_diff,
                confidence,
                model2unitcost,
            )
            self.time_opt_expected += time.time() - start_time
            # if cnt == 100:
            #     print(f"k: {k}, Expected cost: {expected}, Profile cost: {profile_cost}, Apply cost: {apply_cost}")
            if expected < min_expected:
                min_expected = expected
                k_expected = k
            # else:
            #     break
            k *= 2
        return min_expected, k_expected

    def prune_models(
        self,
        cnt,
        nr_items_total,
        model2result,
        model2conform,
        best_model,
        acc_diff,
        confidence,
        is_accumulated=True,
    ):
        # Select models for pruning.
        unknown_models = self.get_profile_models(model2result)
        if not unknown_models or cnt == nr_items_total:
            return
        # Compute unit cost per model.
        model2unitcost = {
            model: self.model2info[model].calculate_cost(
                model2result[model].nr_prompt_tokens,
                model2result[model].nr_completion_tokens,
            )
            / cnt
            for model in model2result
        }
        satisfied_models = [
            model
            for model in self.ordered_models
            if model2result[model].status == ModelStatus.satisfied
        ]
        model_satisfied = satisfied_models[0]
        # Compute cost of using the current smallest satisfied model.
        cost_satified = model2unitcost[model_satisfied] * (nr_items_total - cnt)
        # if self.method == "multi-model":
        #     # Find the best combination based on mixed integer programming.
        #     _, opt_cost, _ = self.find_ratios(
        #         cnt, nr_items_total, model2result, best_model, acc_diff, confidence
        #     )
        #     cost_satified = opt_cost
        # else:
        #     # Compute cost of using the current smallest satisfied model.
        #     cost_satified = model2unitcost[model_satisfied] * (nr_items_total - cnt)
        # Compute expected cost.
        if is_accumulated:
            expected, k = self.min_expected_cost(
                unknown_models,
                model_satisfied,
                cnt,
                nr_items_total,
                model2conform,
                best_model,
                acc_diff,
                confidence,
                model2unitcost,
            )
            if cnt % 1000 == 0 or expected >= cost_satified:
                print(
                    f"Cnt: {cnt}, k: {k}, Expected cost: {expected}, Satisfied cost: {cost_satified}, Conform: {model2conform}"
                )
            if expected >= cost_satified:
                for model in unknown_models:
                    model2result[model].status = ModelStatus.invalid
                    print(f"Removed {model} from valid models.")
        else:
            for model in unknown_models:  
                expected, k = self.min_expected_cost(
                    [model],
                    model_satisfied,
                    cnt,
                    nr_items_total,
                    model2conform,
                    best_model,
                    acc_diff,
                    confidence,
                    model2unitcost,
                )
                if cnt % 1000 == 0 or expected >= cost_satified:
                    print(
                        f"Cnt: {cnt}, k: {k}, Model: {model}, Expected cost: {expected}, Satisfied cost: {cost_satified}, Conform: {model2conform[model]}"
                    )
                if expected >= cost_satified:
                    model2result[model].status = ModelStatus.invalid
                    print(f"Removed {model} from valid models.")

    def prune_models_prev(
        self,
        cnt,
        nr_items_total,
        model2result,
        model2conform,
        best_model,
        acc_diff,
        confidence,
    ):
        # Compute previous cost.
        model2unitcost = {
            model: self.model2info[model].calculate_cost(
                model2result[model].nr_prompt_tokens,
                model2result[model].nr_completion_tokens,
            )
            / cnt
            for model in model2result
        }

        satisfied_models = [
            model
            for model in self.ordered_models
            if model2result[model].status == ModelStatus.satisfied
        ]
        model_satisfied = satisfied_models[0]
        cost_satified = model2unitcost[model_satisfied] * (nr_items_total - cnt)
        # Select models for pruning.
        unknown_models = self.get_profile_models(model2result)
        # Compute expected cost for each unknown model.
        for model in unknown_models:
            if model == best_model:
                continue
            p = model2conform[model] / cnt
            std = (p * (1 - p) / cnt) ** 0.5
            func_cost_model = partial(
                utils_math.func_cost,
                nr_items_total - cnt,
                1 - acc_diff,
                norm.ppf(1 - (1 - confidence) / 2),
                model2unitcost[model],
                model2unitcost[best_model],
                model2unitcost[model_satisfied],
            )
            # When accuracy is one.
            if model2conform[model] == cnt:
                expected = model2unitcost[best_model]
            else:
                start_time = time.time()
                expected = norm.expect(
                    func_cost_model, loc=p, scale=std, lb=0.0, ub=1.0
                )
                self.time_opt_expected += time.time() - start_time

            if cnt % 1000 == 0 or expected >= cost_satified:
                print(
                    f"Cnt: {cnt}, Model: {model}, Expected cost: {expected}, Satisfied cost: {cost_satified}, P: {p}, Std: {std}"
                )
            if expected >= cost_satified:
                model2result[model].status = ModelStatus.invalid
                print(f"Removed {model} from valid models.")

    def exhaustive(self, df, acc_diff, confidence):
        # Find smallest model that satisfies the accuracy requirement.
        smallest_model, cost, cnt, model2result = self.find_smallest_model(
            df, acc_diff, confidence
        )
        # Iterate over the remaining instances.
        result = model2result[smallest_model]
        for idx, row in df.iloc[cnt:].iterrows():
            # nr_prompt_tokens = self.model2info[model].get_nr_tokens(row['text'])
            nr_prompt_tokens = row["nr_prompt_tokens"][smallest_model]
            nr_completion_tokens = row["nr_completion_tokens"][smallest_model]
            result.add_instance(
                idx, row[smallest_model], nr_prompt_tokens, nr_completion_tokens
            )
            cost += self.model2info[smallest_model].calculate_cost(
                nr_prompt_tokens, nr_completion_tokens
            )
        return cost, model2result

    def efficient(self, df, acc_diff, confidence):
        # Process until expected cost is lower than the cost of the largest model.
        smallest_model, cost, cnt, model2result = self.find_smallest_model(
            df, acc_diff, confidence, use_expected=True
        )
        # Iterate over the remaining instances.
        result = model2result[smallest_model]
        for idx, row in df.iloc[cnt:].iterrows():
            # nr_prompt_tokens = self.model2info[model].get_nr_tokens(row['text'])
            nr_prompt_tokens = row["nr_prompt_tokens"][smallest_model]
            nr_completion_tokens = row["nr_completion_tokens"][smallest_model]
            result.add_instance(
                idx, row[smallest_model], nr_prompt_tokens, nr_completion_tokens
            )
            cost += self.model2info[smallest_model].calculate_cost(
                nr_prompt_tokens, nr_completion_tokens
            )
        return cost, model2result

    def multi_model(self, df, acc_diff, confidence):
        best_model = Processor.best_model
        # Find smallest model that satisfies the accuracy requirement.
        _, cost, cnt, model2result = self.find_smallest_model(
            df, acc_diff, confidence, use_expected=True
        )
        if cnt == len(df):
            return cost, model2result
        # Find the best combination based on mixed integer programming.
        model2ratio, opt_cost, opt_combination = self.find_ratios(
            cnt, len(df), model2result, best_model, acc_diff, confidence
        )
        print(
            f"Optimal ratio: {model2ratio}, Optimal cost: {opt_cost}, Optimal combination: {opt_combination}"
        )
        # Process remaining instances based on the ratios.
        cnt_remaining = len(df) - cnt
        process_cnts = utils_math.split_integer_into_parts(
            cnt_remaining, list(model2ratio.values())
        )
        print(f"Remaining cnt: {cnt_remaining}, Process counts: {process_cnts}")
        for model, process_cnt in zip(model2ratio, process_cnts):
            result = model2result[model]
            for idx, row in df.iloc[cnt : (cnt + process_cnt)].iterrows():
                # nr_prompt_tokens = self.model2info[model].get_nr_tokens(row['text'])
                nr_prompt_tokens = row["nr_prompt_tokens"][model]
                nr_completion_tokens = row["nr_completion_tokens"][model]
                result.add_instance(
                    idx, row[model], nr_prompt_tokens, nr_completion_tokens
                )
                cost += self.model2info[model].calculate_cost(
                    nr_prompt_tokens, nr_completion_tokens
                )
            cnt += process_cnt
        assert cnt == len(df)
        return cost, model2result

    def find_ratios(
        self, cnt, nr_items_total, model2result, best_model, acc_diff, confidence
    ):
        # Find the best combination based on mixed integer programming.
        acc_target = (
            (1 - acc_diff - cnt / nr_items_total)
            * nr_items_total
            / (nr_items_total - cnt)
        )
        # Compute lower bound for accuracy for different confidence levels.
        model2conform = {
            model: sum(
                1
                for idx, out in result.idx2out.items()
                if out == model2result[best_model].idx2out[idx]
            )
            for model, result in model2result.items()
            if model != best_model
        }
        model2cnt = {
            model: len(result.idx2out)
            for model, result in model2result.items()
            if model != best_model
        }
        start_time = time.time()
        model2conf2acc = {
            model: {
                temp_confidence: utils_math.binom_ci(
                    nr_conform, model2cnt[model], temp_confidence
                )[0]
                for temp_confidence in np.linspace(
                    confidence, 1.0, num=5, endpoint=False
                )
            }
            for model, nr_conform in model2conform.items()
        }
        self.time_opt_ci += time.time() - start_time
        # Compute unit cost per model.
        model2unitcost = {
            model: self.model2info[model].calculate_cost(
                model2result[model].nr_prompt_tokens,
                model2result[model].nr_completion_tokens,
            )
            / cnt
            for model in model2result
        }
        # Compute costs to process remaining instances.
        model2cost = {
            model: unitcost * (nr_items_total - cnt)
            for model, unitcost in model2unitcost.items()
        }
        # Compute the optimal combination.
        start_time = time.time()
        model2ratio, opt_cost, opt_combination = self.compute_optimal(
            model2conf2acc, model2cost, acc_target, confidence
        )
        self.time_opt_program += time.time() - start_time
        return model2ratio, opt_cost, opt_combination

    def compute_optimal(self, model2conf2acc, model2cost, acc_target, conf_target):
        best_model = Processor.best_model
        # Order costs based on the ordered models.
        costs = [model2cost[model] for model in self.ordered_models]
        # Iterate over all possible combinations of confidence levels.
        conf_acc_lists = [
            # Add a case for not using the model at all, i.e., (1.0, 0.0).
            [(conf, acc) for conf, acc in model2conf2acc[model].items()] + [(1.0, 0.0)]
            if model != best_model
            else [(1.0, 1.0)]
            for model in self.ordered_models
        ]
        # Find optimal cost per combination.
        combination2opt = {}
        for combination in itertools.product(*conf_acc_lists):
            confs, accs = zip(*combination)
            # If the accumulated confidence is lower than the target, skip this combination.
            conf_accumulated = np.prod(confs)
            if conf_accumulated < conf_target:
                continue
            # Coefficients of the objective function
            c = costs
            # Coefficients of the inequality constraints (left-hand side)
            inverted_accs = [-1 * acc for acc in accs]
            A_ub = [inverted_accs]
            # Inequality constraints (right-hand side)
            b_ub = [-acc_target]
            # Equality constraint coefficients (left-hand side)
            A_eq = [[1] * len(self.ordered_models)]
            # Equality constraints (right-hand side)
            b_eq = [1]
            # Solve the problem
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
            # print(combination, res.success, res.fun, res.x)
            if res.success:
                combination2opt[combination] = res

        # Find the combination with the lowest cost.
        opt_combination = min(combination2opt, key=lambda k: combination2opt[k].fun)
        ratios = combination2opt[opt_combination].x
        model2ratio = {
            model: ratio for model, ratio in zip(self.ordered_models, ratios)
        }
        cost = combination2opt[opt_combination].fun
        return model2ratio, cost, opt_combination


class ModelStatus(Enum):
    unknown = 1
    invalid = 2
    satisfied = 3


class ModelResult:
    def __init__(self, model):
        self.model = model
        self.nr_prompt_tokens = 0
        self.nr_completion_tokens = 0
        self.status = ModelStatus.unknown
        self.idx2out = {}

    def add_instance(self, idx, out, nr_prompt_tokens, nr_completion_tokens):
        self.idx2out[idx] = out
        self.nr_prompt_tokens += nr_prompt_tokens
        self.nr_completion_tokens += nr_completion_tokens
