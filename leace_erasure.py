import argparse
from data_selection_erasure import (
    leace_convos_prism,
    leace_convos_cad,
    evaluation_prism,
    evaluation_cad,
)
import datasets
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llama import scrub_llama

from preprocess_data import get_prism_convos, get_cad_convos

from probing import balance_df, get_representations, train_probe


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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
        "-d",
        "--dataset",
        type=str,
        default="prism",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "-item",
        type=str,
        default="",
        help="Item to run evaluation on",
    )
    parser.add_argument(
        "-domain",
        type=str,
        default="",
        help="Evaluation domain to run questions from",
    )
    parser.add_argument(
        "-demographic",
        type=str,
        default="",
        help="Demographic to probe for",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Whether to balance the dataset for the demographic attribute",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_pickle(f"{args.data_folder}/{args.dataset}_preprocessed.gz")

    if args.dataset == "prism":
        convo_func = get_prism_convos
        leace_convos = leace_convos_prism
        leace_df = pd.read_pickle(f"{args.data_folder}/cad_en_preprocessed.gz")
        leace_convo_func = get_cad_convos
        evaluation = evaluation_prism
    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        leace_convos = leace_convos_cad
        leace_df = pd.read_pickle(f"{args.data_folder}/prism_preprocessed.gz")
        leace_convo_func = get_prism_convos
        evaluation = evaluation_cad

    if args.domain:
        df_questions = pd.read_pickle(f"{args.data_folder}/questions.gz")
        qrange = {
            "benefits": (61, 91),
            "political": (91, 121),
            "salary": (121, 151),
            "legal": (151, 181),
            "medical": (181, 211),
        }[args.domain]
        df_questions = df_questions.loc[
            df_questions["q_id"].isin([f"q_{i}" for i in range(*qrange)])
        ]

        eval_convos = [
            c_id
            for group in evaluation[args.item]
            for c_id in evaluation[args.item][group]
        ]
        evaluation_df = df.loc[
            df["conversation_id"].isin(eval_convos)
        ].reset_index(drop=True)
        convos = convo_func(evaluation_df)

    # use leace data
    leace_cs = [
        c_id
        for group in leace_convos[args.item]
        for c_id in leace_convos[args.item][group]
    ]
    selected_leace_df = leace_df.loc[
        leace_df["conversation_id"].isin(leace_cs)
    ]
    reverse_label_dict = {
        c_id: group
        for group in leace_convos[args.item]
        for c_id in leace_convos[args.item][group]
    }
    leace_cs = leace_convo_func(selected_leace_df)

    leace_labels = selected_leace_df["conversation_id"].map(reverse_label_dict)
    inputs = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        for convo in leace_cs
    ]
    labels = [leace_labels.iloc[i] for i in range(len(leace_cs))]

    leace_cs = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        | {"label": leace_labels.iloc[i]}
        for i, convo in enumerate(leace_cs)
    ]
    leace_dataset = datasets.Dataset.from_pandas(pd.DataFrame(leace_cs))
    leace_dataset = leace_dataset.class_encode_column("label")

    with torch.no_grad():
        logits = [
            model(
                inp.to(device),
                do_sample=False,
                max_new_tokens=1,
            )[
                "logits"
            ][-1, -1, :]
            .detach()
            .cpu()
            .clone()
            .to(torch.float)
            for inp in tqdm(inputs)
        ]

    train_ids, test_ids, y_train, y_test = train_test_split(
        range(len(logits)),
        labels,
        test_size=0.33,
        random_state=42,
        stratify=labels,
    )

    lr = LogisticRegression(max_iter=1000).fit(
        np.array(logits)[train_ids], y_train
    )
    beta = torch.from_numpy(lr.coef_)
    print(beta.norm(p=torch.inf))
    print(
        "start score half",
        lr.score(logits[test_ids], y_test),
    )

    scrubber = scrub_llama(
        model,
        leace_dataset,
        z_column="label",
        train_ids=train_ids,
        test_ids=test_ids,
    )
    with scrubber.scrub(model):
        if args.domain:
            leace_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            for row in tqdm(df_questions.itertuples(index=False)):
                convos_and_questions = [
                    tokenizer.apply_chat_template(
                        convo + [{"role": "user", "content": row.question}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for convo in convos
                ]
                tokens = 1
                outputs = [
                    answer[0]["generated_text"]
                    for answer in tqdm(
                        leace_pipeline(
                            ListDataset(convos_and_questions),
                            batch_size=32,
                            max_new_tokens=tokens,
                            return_full_text=False,
                            do_sample=False,
                        )
                    )
                ]

                evaluation_df = pd.concat(
                    [evaluation_df, pd.DataFrame({row.q_id: outputs})],
                    axis=1,
                )
                evaluation_df.to_pickle(
                    f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_leace_{args.domain}_{args.item}_answers.gz"
                )
        elif args.demographic:
            representations = get_representations(
                df, convo_func, tokenizer, model, device
            )
            if args.dataset == "prism":
                if args.balanced:
                    df = balance_df(df, args.demographic, "")
                else:
                    df = df.loc[
                        ~(df[args.demographic].isna())
                        & (df[args.demographic] != "Prefer not to say")
                        & (df[args.demographic] != "Unknown")
                        & (df[args.demographic] != "female-male-non-binary")
                    ]
            elif "cad" in args.dataset:
                if args.balanced:
                    df = balance_df(
                        df, args.demographic, args.dataset.split("_")[-1]
                    )
                else:
                    df = df.loc[
                        ~(df[args.demographic].isna())
                        & (df[args.demographic] != "Prefer not to say")
                        & (df[args.demographic] != "other")
                        & (df[args.demographic] != "Unknown")
                        & (df[args.demographic] != "female-male-non-binary")
                    ]
            elif args.dataset == "chen":
                if args.balanced:
                    df = balance_df(df, args.demographic, "")
                else:
                    df = df.loc[
                        ~(df[args.demographic].isna())
                        & (df[args.demographic] != "Unknown")
                        & (df[args.demographic] != "female-male-non-binary")
                    ]
            representations = representations[df.index]
            scores = train_probe(
                df[args.demographic].tolist(),
                representations,
                args.dataset,
                args.n_layers,
                args.demographic,
                save=args.save,
                save_file=args.results_folder + f"/{args.model.split('/')[1]}",
            )
            with open(
                args.results_folder
                + f"/{args.model.split('/')[1]}_{args.dataset}_leace_{args.demographic.replace(' ','')}{'_balanced' if args.balanced else ''}_{args.item}_scores.pkl",
                "wb",
            ) as outfile:
                pickle.dump(scores, outfile)
