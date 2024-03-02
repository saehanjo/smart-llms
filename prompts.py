prompt_regex = """Based on the following movie review examples and labels for binary sentiment classification, write a python program based on regex that can classify the sentiment of a given text. Output should be one of the three labels: positive, negative, and uncertain if there is not enough information for classification.

EXAMPLES:
"""

prompt_template = """Classify the following movie review as positive or negative based on its sentiment.

Text: {text}
Label: """

prompt_template_01 = """Classify the following movie review as positive or negative based on its sentiment. Label using "1" for positive and "0" for negative.

Text: {text}
Label: """


def get_example_str(df, label_map, nr_examples, add_label):
    # Extract examples and labels.
    examples_str = ""
    if add_label:
        for label_key in label_map:
            # Filter to extract examples with the current label.
            df_label = df[df["label"] == label_key]
            for index, row in df_label.head(nr_examples).iterrows():
                # Add example to the string.
                examples_str += f"Text: {row['text']}\n"
                examples_str += f"Label: {label_map[row['label']]}\n\n"
    else:
        for index, row in df.head(nr_examples).iterrows():
            # Add example to the string.
            examples_str += f"Text: {row['text']}\n"
    return examples_str


def get_prompt_code_generation(df, label_map, nr_examples, add_label):
    examples_str = get_example_str(
        df, label_map, nr_examples, add_label
    )
    prompt_final = prompt_regex + examples_str
    print(prompt_final)
    return prompt_final
