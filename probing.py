import argparse
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_repr(df, model, tokenizer, device):
    inputs = [
        tokenizer.apply_chat_template(
            [
                {"role": turn["role"], "content": turn["content"]}
                for turn in convo
                if turn["role"] == "user" or turn["if_chosen"] == True
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for convo in df["conversation_history"].tolist()
    ]

    representations = [
        [
            rep[-1, -1, :].detach().cpu().clone().to(torch.float)
            for rep in model(
                inp,
                output_hidden_states=True,
                max_new_tokens=1,
                return_dict=True,
            )["hidden_states"]
        ]
        for inp in inputs
    ]

    df["representations"] = representations

    n_layers = len(representations[0])
    return df, n_layers


def train_probe(
    df, n_layers, results_file, demographic_cols, save=False, save_file=""
):
    results = {}
    for demographic_col in tqdm(demographic_cols):
        results[demographic_col] = []
        for l in tqdm(range(n_layers)):
            X = [rep[l] for rep in df["representations"].tolist()]
            y = df[demographic_col].tolist()
            clf = LogisticRegression(random_state=42)

            if save:
                clf = clf.fit(X, y)
                with open(
                    save_file + f"_{demographic_col}_{l}.pkl", "wb"
                ) as outfile:
                    pickle.dump(clf, outfile)
            else:
                scores = cross_val_score(clf, X, y, cv=5)
                results[demographic_col].append(scores)
    if not save:
        with open(
            results_file,
            "wb",
        ) as outfile:
            pickle.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--save_probe", action="store_true", help="Save trained probe"
    )
    # parser.add_argument(
    #     "--random", action="store_true", help="Train probe on random labels"
    # )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device)

    if os.path.isfile(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_preprocessed.gz"
    ):
        df = pd.read_pickle(
            "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_preprocessed.gz",
            compression="gzip",
        )
    else:
        conversations = load_dataset(
            "HannahRoseKirk/prism-alignment", "conversations"
        )["train"].to_pandas()
        survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")[
            "train"
        ].to_pandas()

        df = pd.merge(
            conversations,
            survey,
            on=["user_id"],
        )
        to_simplify = ["religion", "ethnicity"]
        for column in to_simplify:
            df[column] = df[column].apply(
                lambda x: (
                    dict(x)["simplified"]
                    if type(x) == dict
                    else "Prefer not to say"
                )
            )
        regions = ["birth_region", "reside_region"]
        for region in regions:
            df[region] = df["location"].apply(
                lambda x: (
                    dict(x)[region] if type(x) == dict else "Prefer not to say"
                )
            )

        df.to_pickle(
            "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_preprocessed.gz"
        )

    df, n_layers = get_repr(df, model, tokenizer, device)
    df.to_pickle(
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_representations.gz"
    )

    demographic_cols = [
        "age",
        "gender",
        "religion",
        "ethnicity",
        "employment_status",
        "education",
        "birth_region",
        "reside_region",
        "marital_status",  # not in facts paper
        "english_proficiency",  # not in facts paper
    ]

    train_probe(
        reps,
        n_layers,
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_probe_results.pkl",
        demographic_cols,
        # random=args.random,
        save=args.save_probe,
        save_file=f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_probe",
    )
