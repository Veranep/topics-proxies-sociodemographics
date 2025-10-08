import argparse
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

chat_templates = {
    "allenai/OLMo-2-1124-7B": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
}


def get_repr(df, model, tokenizer, device, agg_method, questions):
    convos = [
        [
            {
                "role": turn["role"].replace("model", "assistant"),
                "content": turn["content"],
            }
            for turn in convo
            if turn["role"] == "user" or turn["if_chosen"] == True
        ]
        for convo in df["conversation_history"].tolist()
    ]
    for convo in convos:
        to_remove = []
        for i in range(len(convo)):
            if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for idx in to_remove:
            del convo[idx]
    if questions:
        inputs = [
            tokenizer.apply_chat_template(
                convo + [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            for question in questions
            for convo in convos
        ]
    else:
        inputs = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            for convo in convos
        ]
    if agg_method == "last":
        representations = [
            [
                rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs)
        ]
    elif agg_method == "mean":
        representations = [
            [
                torch.mean(
                    rep[-1, :, :].detach().cpu().clone().to(torch.float), 0
                )
                for rep in model(
                    inp.to(device),
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs)
        ]

    df["representations"] = representations

    return df


def train_probe(
    df, n_layers, results_file, demographic_cols, save=False, save_file=""
):
    results = {}
    for demographic_col in tqdm(demographic_cols):
        results[demographic_col] = []
        for l in tqdm(range(n_layers)):
            X = [rep[l] for rep in df["representations"].tolist()]
            y = df[demographic_col].tolist()
            clf = LogisticRegression(
                random_state=42,
            )

            if save:
                clf = pipeline.fit(X, y)
                with open(
                    save_file + f"_{demographic_col}_{l}.pkl", "wb"
                ) as outfile:
                    pickle.dump(clf, outfile)
            else:
                scores = cross_validate(
                    clf,
                    X,
                    y,
                    cv=5,
                    scoring=[
                        "f1_micro",
                        "f1_macro",
                        "f1_weighted",
                        "balanced_accuracy",
                    ],
                )
                results[demographic_col].append(
                    {
                        "f1_micro": scores["test_f1_micro"],
                        "f1_macro": scores["test_f1_macro"],
                        "f1_weighted": scores["test_f1_weighted"],
                        "balanced_accuracy": scores["test_balanced_accuracy"],
                    }
                )
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
        "-am",
        "--agg_method",
        type=str,
        default="mean",
        help="Method for aggregating representations across a conversation",
    )
    parser.add_argument(
        "--save_probe", action="store_true", help="Save trained probe"
    )
    parser.add_argument(
        "--add_questions", action="store_true", help="Probe after question"
    )
    # parser.add_argument(
    #     "--random", action="store_true", help="Train probe on random labels"
    # )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        tokenizer.chat_template = chat_templates[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if os.path.isfile(
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_representations.gz"
    ):
        df = pd.read_pickle(
            f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_representations.gz",
            compression="gzip",
        )
    else:
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
                        dict(x)[region]
                        if type(x) == dict
                        else "Prefer not to say"
                    )
                )

            df.to_pickle(
                "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_preprocessed.gz"
            )

        questions = list(
            set(
                pd.read_csv("old/medical_llama_prompts.csv")[
                    "prompts"
                ].tolist()
                + pd.read_csv("old/medical_qwen_prompts.csv")[
                    "prompts"
                ].tolist(),
            )
        )

        df = get_repr(
            df,
            model,
            tokenizer,
            device,
            args.agg_method,
            questions=questions if args.add_questions else None,
        )
        # df.to_pickle(
        #     f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_representations.gz"
        # )

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

    n_layers = len(df.iloc[0]["representations"])

    train_probe(
        df,
        n_layers,
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_probe_results_{args.agg_method}.pkl",
        demographic_cols,
        # random=args.random,
        save=args.save_probe,
        save_file=f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_probe",
    )
