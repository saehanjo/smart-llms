import tiktoken

import utils_llm


class ModelInfo:
    def __init__(self, model, prompt_cost, completion_cost):
        self.model = model
        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost

    def calculate_cost(self, nr_prompt_tokens, nr_completion_tokens):
        prompt_cost = nr_prompt_tokens * self.prompt_cost / 1000
        completion_cost = nr_completion_tokens * self.completion_cost / 1000
        total_cost = prompt_cost + completion_cost
        return total_cost

    def get_nr_tokens(self, text):
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    def is_chat(self):
        return utils_llm.check_if_chat(self.model)


def get_model2info():
    model2info = {
        model: ModelInfo(model, pricing["prompt"], pricing["completion"])
        for model, pricing in utils_llm.model2pricing.items()
    }
    return model2info
