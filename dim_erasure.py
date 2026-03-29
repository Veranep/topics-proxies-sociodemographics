import argparse
import datasets
from torch.utils.data import Dataset
import torch.nn as nn
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

from preprocess_data import get_prism_convos, get_cad_convos

from probing import balance_df, get_representations, train_probe

# Inspiration: https://github.com/fholstege/uncensoringllms/tree/main


def get_example_ids(df, item, is_prism):
    all_ids = df["conversation_id"].unique()
    if "topic" in item:
        keywords = item.split(":")[1].split(",")
        if is_prism:
            in_ids = df[df["topic"] == keywords[0]]["conversation_id"]
        else:
            in_ids = []
            print(keywords)
            for k in keywords:
                in_ids += df[df["topic"].str.contains(f" {k} ", na=False)][
                    "conversation_id"
                ].tolist()
        out_ids = [cid for cid in all_ids if cid not in in_ids]
        np.random.shuffle(in_ids)
        np.random.shuffle(out_ids)
        in_ids = in_ids[:200]
        out_ids = out_ids[:200]
    else:
        if item not in df.columns:
            raise Exception(f"Column {item} is not in the data")
        df = (
            df.groupby(["conversation_id", "topic"])[
                [
                    c
                    for c in df.columns
                    if ("model_response" in c or "user_prompt" in c)
                    and (c not in ["model_response", "user_prompt"])
                ]
            ]
            .mean()
            .reset_index()
        )
        df = df.sort_values(by=item)
        in_ids = df.iloc[:200]["conversation_id"].tolist()
        out_ids = df.iloc[-200:]["conversation_id"].tolist()
    return in_ids, out_ids


def get_model_layers(model):
    """Helper function to get model layers based on architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA, Mistral, etc.
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # Qwen, GPT-2, etc.
        return model.transformer.h
    elif hasattr(model, "transformer") and hasattr(
        model.transformer, "layers"
    ):
        # Some other architectures
        return model.transformer.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        # GPT-NeoX based models
        return model.gpt_neox.layers
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        # Some encoder-decoder architectures
        return model.model.decoder.layers
    else:
        # Try to find layers using more general pattern matching
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, "layers"):
                return attr.layers
            if hasattr(attr, "h") and isinstance(
                getattr(attr, "h"), torch.nn.ModuleList
            ):
                return attr.h

        raise AttributeError(
            f"Could not find layers in model of type {type(model).__name__}. "
            "Please specify the correct attribute path to access layers."
        )


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class AblationDecoderLayer(nn.Module):
    def __init__(self, original_layer, direction):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer

        # Store the direction in the correct device and dtype upfront
        self.r = direction.T

        # take the unit
        self.r_unit = self.r / torch.norm(self.r)

        self.projection = torch.matmul(self.r_unit, self.r_unit.T)

    def forward(self, *args, **kwargs):
        # get the hidden states
        hidden_states = args[0]

        projection = self.projection.to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        # apply the projection to all the hidden states
        proj = torch.matmul(hidden_states, projection)  # self.

        # remove the projection
        ablated = hidden_states - proj

        # apply to the first argument
        args = (ablated,) + args[1:]

        # return the forward pass of the original layer
        return self.original_layer.forward(*args, **kwargs)


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
        "-n",
        "--n_layers",
        type=int,
        default=None,
        help="Number of model layers",
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
        dim_dataset = "cad_en"
        dim_convo_func = get_cad_convos

    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        dim_dataset = "prism"
        dim_convo_func = get_prism_convos

    dim_df = pd.read_pickle(
        f"{args.data_folder}/{dim_dataset}_preprocessed.gz"
    )
    dim_linguistic_df = pd.read_pickle(
        f"data/{dim_dataset}_utterances_linguistic.gz"
    ).drop(
        columns=["s_neutral_model_response", "s_neutral_user_prompt"],
        errors="ignore",
    )
    for c in ["politeness_user_prompt", "politeness_model_response"]:
        if c in dim_linguistic_df:
            dim_linguistic_df[c] = dim_linguistic_df[c].replace(
                {
                    "impolite": 0,
                    "neutral": 0.5,
                    "polite": 1,
                    "somewhat polite": 0.75,
                }
            )
    dim_linguistic_df = dim_linguistic_df.rename(
        columns={"gpt_description": "topic"}
    )

    in_ids, out_ids = get_example_ids(
        dim_linguistic_df, args.item, "cad" in args.dataset
    )

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

        convos = convo_func(df)

    dim_in_convos = dim_convo_func(
        dim_df[dim_df["conversation_id"].isin(in_ids)]
    )
    dim_out_convos = dim_convo_func(
        dim_df[dim_df["conversation_id"].isin(out_ids)]
    )

    inputs_0 = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        for convo in dim_in_convos
    ]
    inputs_1 = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        for convo in dim_out_convos
    ]

    representations_0 = np.array(
        [
            [
                rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs_0)
        ]
    )
    representations_1 = np.array(
        [
            [
                rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"]
            ]
            for inp in tqdm(inputs_1)
        ]
    )

    layers = get_model_layers(model)
    for layer_idx in range(1, args.n_layers):

        # Extract hidden states at the specified layer and position
        hidden_0 = [torch.tensor(r[layer_idx]) for r in representations_0]
        hidden_1 = [torch.tensor(r[layer_idx]) for r in representations_1]

        if layer_idx == (args.n_layers - 1):
            train_ids, test_ids, y_train, y_test = train_test_split(
                range(len(hidden_0 + hidden_1)),
                [0] * len(hidden_0) + [1] * len(hidden_1),
                test_size=0.33,
                random_state=42,
                stratify=[0] * len(hidden_0) + [1] * len(hidden_1),
            )
            lr = LogisticRegression(max_iter=1000).fit(
                np.array(hidden_0 + hidden_1)[train_ids],
                y_train,
            )
            beta = torch.from_numpy(lr.coef_)
            print(beta.norm(p=torch.inf))
            print(
                "start score half",
                lr.score(
                    np.array(hidden_0 + hidden_1)[test_ids],
                    y_test,
                ),
            )

        # Compute mean of hidden states for each category
        mean_0 = torch.stack(hidden_0).mean(dim=0)
        mean_1 = torch.stack(hidden_1).mean(dim=0)

        # Compute refusal direction as the normalized difference between harmful and harmless means
        concept_dir = mean_0 - mean_1
        concept_dir = concept_dir / concept_dir.norm()
        concept_dir = concept_dir.to(device)
        model.model.layers[layer_idx] = AblationDecoderLayer(
            layers[layer_idx], concept_dir.unsqueeze(dim=0)
        )

    with torch.no_grad():
        after_0 = [
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
            for inp in tqdm(inputs_0)
        ]
        after_1 = [
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
            for inp in tqdm(inputs_1)
        ]

    lr = LogisticRegression(max_iter=1000).fit(
        np.array(after_0 + after_1)[train_ids], y_train
    )
    beta = torch.from_numpy(lr.coef_)
    print(beta.norm(p=torch.inf))
    print(
        "end score half",
        lr.score(np.array(after_0 + after_1)[test_ids], y_test),
    )

    if args.domain:
        dim_pipeline = pipeline(
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
                    dim_pipeline(
                        ListDataset(convos_and_questions),
                        batch_size=32,
                        max_new_tokens=tokens,
                        return_full_text=False,
                        do_sample=False,
                    )
                )
            ]

            df = pd.concat(
                [df, pd.DataFrame({row.q_id: outputs})],
                axis=1,
            )
            df.to_pickle(
                f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_dim_{args.domain}_{args.item}_answers.gz"
            )
    elif args.demographic:
        representations = get_representations(
            df, convo_func, tokenizer, model, device
        )
        if args.dataset == "prism":
            non_balanced_df = df.loc[
                ~(df[args.demographic].isna())
                & (df[args.demographic] != "Prefer not to say")
                & (df[args.demographic] != "Unknown")
                & (df[args.demographic] != "female-male-non-binary")
            ]
            balanced_df = balance_df(df, args.demographic, "")

        elif "cad" in args.dataset:
            non_balanced_df = df.loc[
                ~(df[args.demographic].isna())
                & (df[args.demographic] != "Prefer not to say")
                & (df[args.demographic] != "other")
                & (df[args.demographic] != "Unknown")
                & (df[args.demographic] != "female-male-non-binary")
            ]
            balanced_df = balance_df(
                df, args.demographic, args.dataset.split("_")[-1]
            )

        for specific_df, specific_label in [
            (non_balanced_df, ""),
            (balanced_df, "balanced"),
        ]:
            if os.path.isfile(
                args.results_folder
                + f"/{args.model.split('/')[1]}_{args.dataset}_dim_{args.demographic.replace(' ','')}{specific_label}_{args.item}_mlp_scores.pkl"
            ):
                pass
            specific_representations = representations[specific_df.index]
            scores = train_probe(
                specific_df[args.demographic].tolist(),
                specific_representations,
                args.dataset,
                args.n_layers,
                args.demographic,
                save=False,
                save_file=args.results_folder + f"/{args.model.split('/')[1]}",
                mlp=True,
            )
            with open(
                args.results_folder
                + f"/{args.model.split('/')[1]}_{args.dataset}_dim_{args.demographic.replace(' ','')}{specific_label}_{args.item}_mlp_scores.pkl",
                "wb",
            ) as outfile:
                pickle.dump(scores, outfile)
