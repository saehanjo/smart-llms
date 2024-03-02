import pandas as pd
import numpy as np
import random


class Simulator:
    def __init__(self, model2conformity, nr_prompt_tokens, nr_completion_tokens):
        self.model2conformity = model2conformity
        self.nr_prompt_tokens = nr_prompt_tokens
        self.nr_completion_tokens = nr_completion_tokens

    def generate_test(self, n, seed=None):
        # Set the random seed for reproducibility.
        rng = np.random.default_rng(seed)
        # Create an empty DataFrame with n rows and columns based on the keys of the dict.
        cols = list(self.model2conformity.keys()) + ['nr_tokens']
        df = pd.DataFrame(columns=cols, index=range(n))
        # For each column, assign 1 or 0 based on the specified probability.
        for model, conformity in self.model2conformity.items():
            df[model] = rng.choice([1, 0], size=n, p=[conformity, 1 - conformity])
        # Assign remaining columns.
        # df['gpt-4'] = 1
        assert (df['gpt-4'] == 1).all()
        df['nr_prompt_tokens'] = [self.nr_prompt_tokens for _ in range(len(df))]
        df['nr_completion_tokens'] = [self.nr_completion_tokens for _ in range(len(df))]
        return df

