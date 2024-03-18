import numpy as np
import openai
from openai import error
import time
import tiktoken


model2pricing = {
    'gpt-4': { # gpt-4-0613
        'prompt': 0.03,
        'completion': 0.06,
    },
    'gpt-3.5-turbo-1106': {
        'prompt': 0.001,
        'completion': 0.002,
    },
    'babbage-002': {
        'prompt': 0.0004,
        'completion': 0.0004,
    },
    'davinci-002': {
        'prompt': 0.002,
        'completion': 0.002,
    },
    'gpt-3.5-turbo-instruct': {
        'prompt': 0.0015,
        'completion': 0.002,
    },
}


def check_if_chat(gpt_model):
    return gpt_model not in ['gpt-3.5-turbo-instruct', 'davinci-002', 'babbage-002']


def get_nr_tokens(text, model):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def calculate_cost(nr_prompt_tokens, nr_completion_tokens, gpt_model):
    if gpt_model not in model2pricing:
        raise ValueError("Invalid model specified")

    pricing = model2pricing[gpt_model]
    prompt_cost = nr_prompt_tokens * pricing["prompt"] / 1000
    completion_cost = nr_completion_tokens * pricing["completion"] / 1000
    total_cost = prompt_cost + completion_cost
    return total_cost


def call_llm(prompt, gpt_model, logit_bias=None, max_tokens=None, nr_max_timeouts=3):
    is_chat = check_if_chat(gpt_model)

    success = False
    timeout_count = 0
    while not success and timeout_count < nr_max_timeouts:
        timeout_count += 1
        try:
            if gpt_model == "gpt-4" and timeout_count > 1:
                time.sleep(5)
            if is_chat:
                response = openai.ChatCompletion.create(
                    model=gpt_model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    request_timeout=30,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                )
            else:
                response = openai.Completion.create(
                    model=gpt_model,
                    temperature=0,
                    prompt=prompt,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                )
            success = True
        except error.Timeout as e:
            print(f"Timeout error: {e}")

    nr_prompt_tokens = response["usage"]["prompt_tokens"]
    nr_completion_tokens = response["usage"]["completion_tokens"]
    if is_chat:
        output = response["choices"][0]["message"]["content"]
    else:
        output = response["choices"][0]["text"]
    return output, nr_prompt_tokens, nr_completion_tokens


def run_llm_one(text, prompt_template, gpt_model, force_output, nr_max_timeouts=3):
    is_chat = check_if_chat(gpt_model)

    if force_output:
        if is_chat:
            logit_bias = {"31587": 100, "43324": 100}
        else:
            logit_bias = {"24561": 100, "31591": 100}
    else:
        logit_bias = None

    prompt = prompt_template.format(text=text)
    output, nr_prompt_tokens, nr_completion_tokens = call_llm(
        prompt, gpt_model, logit_bias=logit_bias, max_tokens=1, nr_max_timeouts=nr_max_timeouts
    )
    predicted = output.lower().strip()
    return predicted, nr_prompt_tokens, nr_completion_tokens


def run_llm(
    df, prompt_template, gpt_model, force_output, label_map=None, output_accuracy=False
):
    if output_accuracy:
        nr_correct = 0
    predictions = []
    nr_prompt_tokens = 0
    nr_completion_tokens = 0
    for idx, row in df.iterrows():
        if gpt_model == "gpt-4":
            time.sleep(5)
        prediction, temp_nr_prompt_tokens, temp_nr_completion_tokens = run_llm_one(
            row["text"], prompt_template, gpt_model, force_output
        )
        predictions.append(prediction)
        nr_prompt_tokens += temp_nr_prompt_tokens
        nr_completion_tokens += temp_nr_completion_tokens
        print(f"Index: {idx}")
        print(f"Text: {row['text']}")
        print(f"Prediction: {prediction}")
        if label_map and "label" in row:
            label = label_map[row["label"]]
            print(f"Label: {label}")
            print(f"Correct: {prediction == label}")
        if output_accuracy and prediction == label:
            nr_correct += 1
    if output_accuracy:
        accuracy = nr_correct / len(df)
        print(f"Accuracy: {accuracy}")
    cost = calculate_cost(nr_prompt_tokens, nr_completion_tokens, gpt_model)
    print(f"Prompt tokens: {nr_prompt_tokens}")
    print(f"Completion tokens: {nr_completion_tokens}")
    print(f"Cost: {cost}")
    if output_accuracy:
        return np.array(predictions), nr_prompt_tokens, nr_completion_tokens, accuracy
    else:
        return np.array(predictions), nr_prompt_tokens, nr_completion_tokens, cost
