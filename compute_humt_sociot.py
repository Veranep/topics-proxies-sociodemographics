import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import torch.nn.functional as F
import argparse
import scipy
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import sys

# Phrase sets
category_terms = {
    "humt": ["He", "She", "It"],
    "sociot_status": [
        "commanded",
        "proclaimed",
        "demanded",
        "pleaded",
        "mentioned",
        "asked",
    ],
    "sociot_social_distance": [
        "friend",
        "partner",
        "girlfriend",
        "boyfriend",
        "husband",
        "wife",
        "stranger",
    ],
    "sociot_gender": ["She", "He"],
    "sociot_warmth": [
        "friend",
        "lover",
        "mentor",
        "idol",
        "stranger",
        "enemy",
        "examiner",
        "dictator",
    ],
}

# Dictionary of length of D+ phrase set for each dimension
category_index = {
    "sociot_warmth": 4,
    "sociot_status": 3,
    "humt": 2,
    "sociot_gender": 1,
    "sociot_social_distance": 6,
}

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).train()


def compute_set_log_sum_exp(inputs_batch):
    # Compute log-sum-exp using vectorized model calls for a batch of inputs.
    with torch.no_grad():
        outputs = model(**inputs_batch, labels=inputs_batch["input_ids"])
        log_probs = -outputs.loss  # Negative loss gives log probabilities
    return torch.logsumexp(log_probs, dim=0).item()


def compute_ci(prompt, candidates, n_samples=100):
    prompt = f'"{prompt}"' if prompt else ""
    input_batches = []

    # Tokenize all inputs once and store in a list
    for word in candidates:
        input_prompt = (
            f"{word} said, {prompt}"
            if word in ["He", "She", "It"]
            else (
                f"He {word}, {prompt}"
                if word.endswith("ed")
                else (
                    f"My {word} said, {prompt}"
                    if word
                    in [
                        "friend",
                        "partner",
                        "girlfriend",
                        "boyfriend",
                        "husband",
                        "wife",
                    ]
                    else f"The {word} said, {prompt}"
                )
            )
        )
        inputs = tokenizer(
            input_prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(device)
        input_batches.append(inputs)
    # Pad each input batch to the same length
    input_ids = pad_sequence(
        [inp["input_ids"].squeeze(0) for inp in input_batches],
        batch_first=True,
    )
    attention_mask = pad_sequence(
        [inp["attention_mask"].squeeze(0) for inp in input_batches],
        batch_first=True,
    )

    # Combine inputs into a single batch
    inputs_batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Process each batch multiple times to collect results
    results = []
    for _ in range(n_samples):
        # Concatenate all inputs into a single batch for vectorized processing
        results.append(compute_set_log_sum_exp(inputs_batch))

    mean_result = np.mean(results)
    sem_result = scipy.stats.sem(results)
    return mean_result, sem_result


def calculate_td(df, input_col, metric):
    # Calculate log-ratios for each role and category in a vectorized manner.
    df = df.dropna(subset=[input_col])
    df[f"{input_col}_trunc"] = df[input_col].str[:300]

    category = metric
    word_list = category_terms[category]
    split_index = category_index[category]
    first_lex, second_lex = word_list[:split_index], word_list[split_index:]

    def compute_row(row):
        response = row[f"{input_col}_trunc"]
        a_score, a_std = compute_ci(response, first_lex)
        b_score, b_std = compute_ci(response, second_lex)
        log_ratio = a_score - b_score
        return log_ratio, np.sqrt(a_std**2 + b_std**2)

    df[[f"{category}_{input_col}", f"std_{category}_{input_col}"]] = df.apply(
        compute_row, axis=1, result_type="expand"
    )
    return df


def main():
    # Custom validation function for multiple metrics
    def validate_metrics(values):
        invalid_metrics = [m for m in values if m not in category_terms]

        if invalid_metrics:
            print(
                f"Error: The following metrics are invalid: {', '.join(invalid_metrics)}\n"
            )
            print("Valid metric choices:")
            for key, values in category_terms.items():
                print(f"  {key}")
            sys.exit(1)

        return values

    # Generate list of valid metric keys for the help message
    valid_metric_keys = ", ".join(category_terms.keys())

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Process a file and input column."
    )
    parser.add_argument(
        "filename", type=str, help="The name of the file to process"
    )
    parser.add_argument(
        "--input_column",
        type=str,
        required=True,
        help="The name of the input column",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        required=True,
        help=f"One or more metric keys to use. Valid options: {valid_metric_keys}",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=f"Output file to save results in. If not specified, will be saved into a new file.",
    )

    args = parser.parse_args()

    args.metric = validate_metrics(args.metric)

    print(f"Processing file: {args.filename}")
    print(
        f"Input column: {args.input_column if args.input_column else 'None'}"
    )
    print(f"Selected metrics:")
    for metric in args.metric:
        print(f"  {metric}")

    try:
        df = pd.read_csv(args.filename)
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found.")
        return

    filename = args.filename
    fil = filename.split("/")[-1].split(".")[0]
    for metric in args.metric:
        print(
            f"Computing {metric} on {args.input_column} column in {filename}"
        )
        df = calculate_td(df, args.input_column, metric)
    # Ensure the results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory {results_dir} to store outputs")

    save_path = "results/%s_%s_%s.csv" % (
        filename.split("/")[-1].split(".")[0],
        args.input_column,
        "_".join(args.metric),
    )
    df.to_csv(save_path)
    print(f"Saved output to {save_path}")


if __name__ == "__main__":
    main()
