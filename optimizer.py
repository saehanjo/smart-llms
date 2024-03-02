from scipy.optimize import linprog
from scipy.stats import binomtest
import time

import prompts
import utils_llm


class Optimizer:
    k_accuracy = 10
    nr_tokens_code = 500
    nr_examples_code = 5
    add_label_code = False

    random_seed = 0
    gpt_models = list(utils_llm.model2pricing.keys())

    def __init__(self, df, label_map, key_map, prompt_template):
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        self.df = df.rename(columns=key_map)
        self.label_map = label_map
        self.prompt_template = prompt_template
        self.accumulated_cost = 0

    def run_bernoulli(self, accuracy_diff_limit, confidence_level, extra_cost_limit=None):
        df = self.df.head(100)

        gpt2predictions = {}
        valid_gpt_models = self.gpt_models.copy()
        current_best = 'gpt-4'
        extra_cost = 0
        for idx, row in df.iterrows():
            for gpt_model in valid_gpt_models:
                if gpt_model == 'gpt-4':
                    time.sleep(1)
                (
                    prediction,
                    nr_prompt_tokens,
                    nr_completion_tokens,
                ) = utils_llm.run_llm_one(
                    row["text"], self.prompt_template, gpt_model, force_output=True
                )
                gpt2predictions.setdefault(gpt_model, []).append(prediction)
                cost = utils_llm.calculate_cost(
                    nr_prompt_tokens, nr_completion_tokens, gpt_model
                )
                if gpt_model != 'gpt-4':
                    extra_cost += cost

            gpt2correct = {
                gpt_model: sum(
                    1
                    for x, y in zip(
                        gpt2predictions['gpt-4'], gpt2predictions[gpt_model]
                    )
                    if x == y
                )
                for gpt_model in valid_gpt_models
                if gpt_model != 'gpt-4'
            }
            gpt2ci = {
                gpt_model: binomtest(nr_correct, idx + 1).proportion_ci(confidence_level=confidence_level)
                for gpt_model, nr_correct in gpt2correct.items()
            }
            for gpt_model, ci in gpt2ci.items():
                print(
                    f"Idx: {idx}, Model: {gpt_model}, Correct: {gpt2correct[gpt_model]}, Low: {ci.low}, High: {ci.high}"
                )
                if ci.high < 1 - accuracy_diff_limit:
                    valid_gpt_models.remove(gpt_model)
                    print(f"Removed {gpt_model} from valid models.")
                    print(f"Valid models: {valid_gpt_models}")
                elif ci.low >= 1 - accuracy_diff_limit:
                    print(f"Start using {gpt_model} for all examples.")

    def collect_cost_info(self):
        self.gpt2nr_tokens = {
            gpt_model: self.df["text"]
            .apply(lambda x: utils_llm.get_nr_tokens(x, gpt_model))
            .sum()
            for gpt_model in self.gpt_models
        }
        print(self.gpt2nr_tokens)
        self.gpt2text_cost = {
            gpt_model: utils_llm.calculate_cost(nr_tokens, 0, gpt_model)
            for gpt_model, nr_tokens in self.gpt2nr_tokens.items()
        }
        print(self.gpt2text_cost)
        self.gpt2prompt_cost = {
            gpt_model: utils_llm.calculate_cost(
                utils_llm.get_nr_tokens(self.prompt_template, gpt_model), 0, gpt_model
            )
            for gpt_model in self.gpt_models
        }
        print(self.gpt2prompt_cost)
        self.gpt2completion_cost = {
            gpt_model: utils_llm.calculate_cost(0, 1, gpt_model)
            for gpt_model in self.gpt_models
        }
        print(self.gpt2completion_cost)
        self.gpt2total_cost = {
            gpt_model: self.gpt2text_cost[gpt_model]
            + len(self.df)
            * (self.gpt2prompt_cost[gpt_model] + self.gpt2completion_cost[gpt_model])
            for gpt_model in self.gpt_models
        }
        print(self.gpt2total_cost)
        prompt_code_generation = prompts.get_prompt_code_generation(
            self.df, self.label_map, self.nr_examples_code, self.add_label_code
        )
        self.gpt2code_cost = {
            gpt_model: utils_llm.calculate_cost(
                utils_llm.get_nr_tokens(prompt_code_generation, gpt_model),
                self.nr_tokens_code,
                gpt_model,
            )
            for gpt_model in self.gpt_models
        }
        print(self.gpt2code_cost)

    def collect_accuracy_info(self):
        df_sample = self.df.head(self.k_accuracy)
        gpt2predictions = {}
        for gpt_model in self.gpt_models:
            (
                predictions,
                nr_prompt_tokens,
                nr_completion_tokens,
                cost,
            ) = utils_llm.run_llm(
                df_sample,
                self.prompt_template,
                gpt_model,
                force_output=True,
                label_map=self.label_map,
            )
            gpt2predictions[gpt_model] = predictions
            self.accumulated_cost += cost

        gpt4_results = gpt2predictions["gpt-4"]
        self.gpt2accuracy = {
            gpt_model: sum(gpt4_results == gpt2predictions[gpt_model]) / self.k_accuracy
            for gpt_model in self.gpt_models
        }
        print(self.gpt2accuracy)

    def compute_optimal(self, cost_limit):
        accuracies = list(self.gpt2accuracy.values())
        accuracies = [
            -1 * accuracy for accuracy in accuracies
        ]  # Negate the accuracies for maximization
        costs = list(self.gpt2total_cost.values())

        # Coefficients of the objective function
        c = accuracies
        # Coefficients of the inequality constraints (left-hand side)
        A_ub = [costs]
        # Inequality constraints (right-hand side)
        b_ub = [cost_limit]
        # Equality constraint coefficients (left-hand side)
        A_eq = [[1] * len(self.gpt2accuracy)]
        # Equality constraints (right-hand side)
        b_eq = [1]
        # Solve the problem
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        # print(res.x, -1 * res.fun, res.success)
        if not res.success:
            raise Exception("Optimization failed.")

        ratios = res.x
        opt_accuracy = -1 * res.fun
        return ratios, opt_accuracy

    def compute_regret(self, cost_limit):
        ratios, opt_accuracy = self.compute_optimal(cost_limit)

        gpt2regret = {}
        for gpt_model in self.gpt_models:
            cost_code = self.gpt2code_cost[gpt_model]
            ratios, accuracy = self.compute_optimal(cost_limit - cost_code)
            regret = opt_accuracy - accuracy
            gpt2regret[gpt_model] = regret

        return gpt2regret
