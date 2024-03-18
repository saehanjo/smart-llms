import pandas as pd
import time

from model_info import get_model2info
from processor import Processor
from simulator import Simulator


def run_experiment(nr_trials=10):
    methods = [
        'naive',
        'exhaustive',
        'efficient',
        'multi-model',
    ]
    acc_diffs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    # Load model info.
    model2info = get_model2info()

    # Load dataset info.
    df = pd.read_csv("llm_result.csv")
    datasets = df['dataset'].unique().tolist()

    # Collect information per dataset.
    dataset2info = {}
    for dataset in datasets:
        dataset2info[dataset] = {}
        dataset2info[dataset]['nr_items'] = df[df['dataset'] == dataset].iloc[0]['nr_items']
        dataset2info[dataset]['conformity'] = {
            row['model']: row['conformity']
            for idx, row in df[df['dataset'] == dataset].iterrows()
            if row['model'] in model2info
        }
        dataset2info[dataset]['nr_prompt_tokens'] = {
            row['model']: row['nr_prompt_tokens'] / row['nr_processed']
            for idx, row in df[df['dataset'] == dataset].iterrows()
            if row['model'] in model2info
        }
        dataset2info[dataset]['nr_completion_tokens'] = {
            row['model']: row['nr_completion_tokens'] / row['nr_processed']
            for idx, row in df[df['dataset'] == dataset].iterrows()
            if row['model'] in model2info
        }
    print(dataset2info)

    rows_method = []
    for dataset in datasets:
        model2conformity = dataset2info[dataset]['conformity']
        print(f"Conformity: {model2conformity}")
        nr_items = dataset2info[dataset]['nr_items']
        # Dictionary of model to nr_prompt_tokens and nr_completion_tokens.
        nr_prompt_tokens = dataset2info[dataset]['nr_prompt_tokens']
        nr_completion_tokens = dataset2info[dataset]['nr_completion_tokens']
        for acc_diff in acc_diffs:
            print(f'Acc diff: {acc_diff}')
            simulator = Simulator(model2conformity, nr_prompt_tokens, nr_completion_tokens)
            for seed in range(nr_trials):
                df_sim = simulator.generate_test(n=nr_items, seed=seed)

                for method in methods:
                    print(f"Method: {method}")
                    processor = Processor(method=method, model2info=model2info)
                    start_time = time.time()
                    cost, model2result = processor.process(
                        df_sim, acc_diff=acc_diff, confidence=0.95
                    )
                    runtime = time.time() - start_time
                    print(f"Total cost: {cost}")
                    print(f"Runtime: {runtime}")
                    # Conformity.
                    best_model = Processor.best_model
                    conform = 0
                    for idx in range(len(df_sim)):
                        if idx in model2result[best_model].idx2out:
                            conform += 1
                        else:
                            # Check if only one of the models has the idx.
                            assert sum(1 for result in model2result.values() if idx in result.idx2out) == 1
                            for model, result in model2result.items():
                                if idx in result.idx2out:
                                    # If 1 then conform else 0 then non-conform.
                                    conform += result.idx2out[idx]
                                    break
                    final_conformity = conform / len(df_sim)
                    print(f"Final conformity: {final_conformity}")
                    acc_satisfy = 1 - acc_diff <= final_conformity
                    rows_method.append(
                        [
                            dataset,
                            acc_diff,
                            model2conformity,
                            method,
                            acc_satisfy,
                            final_conformity,
                            cost,
                            runtime,
                            processor.time_opt_ci,
                            processor.time_opt_expected,
                            processor.time_opt_program,
                            seed,
                        ]
                    )
                    # Cost per model.
                    rows_model = []
                    for model, result in model2result.items():
                        rows_model.append(
                            [
                                model,
                                len(result.idx2out),
                                model2info[model].calculate_cost(
                                    result.nr_prompt_tokens, result.nr_completion_tokens
                                ),
                            ]
                        )
                    df_model = pd.DataFrame(
                        rows_model, columns=['model', 'nr_processed', 'cost']
                    )
                    print(df_model)

    df_method = pd.DataFrame(
        rows_method,
        columns=[
            'dataset',
            'acc_diff',
            'conformity',
            'method',
            'acc_satisfy',
            'final_conformity',
            'cost',
            'runtime',
            'time_opt_ci',
            'time_opt_expected',
            'time_opt_program',
            'seed',
        ],
    )
    print(df_method)
    df_method.to_csv('result/all.csv')


if __name__ == "__main__":
    run_experiment()
