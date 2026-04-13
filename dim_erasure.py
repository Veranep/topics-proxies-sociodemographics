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


def get_example_ids(df, item, item2):
    n = 10
    # n = 200
    all_ids = df["conversation_id"].unique()
    if item == "random":
        ids = np.random.choice(all_ids, n * 2, replace=False)
        in_ids = ids[:n]
        out_ids = ids[-n:]
    elif "topic" in item:
        keywords1 = item.split(":")[1].split(",")
        keywords2 = item2.split(":")[1].split(",")
        if "," in item:
            in_ids = []
            for k in keywords1:
                in_ids += df[df["topic"].str.contains(f" {k} ", na=False)][
                    "conversation_id"
                ].tolist()
            out_ids = []
            for k in keywords2:
                out_ids += df[df["topic"].str.contains(f" {k} ", na=False)][
                    "conversation_id"
                ].tolist()
        else:
            in_ids = df[df["topic"] == f'"{keywords1[0]}"'][
                "conversation_id"
            ].tolist()
            out_ids = df[df["topic"] == f'"{keywords2[0]}"'][
                "conversation_id"
            ].tolist()
        # out_ids = [cid for cid in all_ids if cid not in in_ids]
        np.random.shuffle(in_ids)
        np.random.shuffle(out_ids)
        in_ids = in_ids[:n]
        out_ids = out_ids[:n]
    elif "demographic" in item:
        df, col = binarize_df(df, item.split(":")[1])
        df = df.sort_values(by=col)
        in_ids = df.iloc[:n]["conversation_id"].tolist()
        out_ids = df.iloc[-n:]["conversation_id"].tolist()
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
        in_ids = df.iloc[:n]["conversation_id"].tolist()
        out_ids = df.iloc[-n:]["conversation_id"].tolist()
    return in_ids, out_ids


def binarize_df(df, col):
    if col in ["unknown_token_Age", "value_JSON_Age", "shared_extracted_Age"]:
        df.loc[df[col] == "Young adult", col] = 0
        df.loc[df[col] == "Older adult", col] = 1
    elif col in [
        "unknown_token_Gender",
        "value_JSON_Gender",
        "shared_extracted_Gender",
        "human_Gender",
    ]:
        df.loc[df[col] == "Female", col] = 0
        df.loc[df[col] == "Male", col] = 1
    elif col in [
        "unknown_token_English proficiency",
        "value_JSON_English proficiency",
        "shared_extracted_English proficiency",
    ]:
        df.loc[df[col] == "Native speaker", col] = 0
        df.loc[df[col] == "Non-Native speaker", col] = 1
    elif col in [
        "unknown_token_Ethnicity",
        "value_JSON_Ethnicity",
        "shared_extracted_Ethnicity",
    ]:
        df.loc[df[col] == "White", col] = 0
        df.loc[df[col] == "Asian", col] = 1
    elif col in [
        "unknown_token_Socioeconomic Status",
        "value_JSON_Socioeconomic Status",
        "shared_extracted_Socioeconomic Status",
    ]:
        df.loc[df[col] == "High income", col] = 0
        df.loc[df[col] == "Low income", col] = 1
    elif col in [
        "unknown_token_Educational Background",
        "value_JSON_Educational Background",
        "shared_extracted_Educational Background",
    ]:
        df.loc[df[col] == "High", col] = 0
        df.loc[df[col] == "Low", col] = 1
    elif col in [
        "unknown_token_Marital Status",
        "value_JSON_Marital Status",
        "shared_extracted_Marital Status",
    ]:
        df.loc[df[col] == "Married", col] = 0
        df.loc[df[col] == "Never married", col] = 1
    elif col in [
        "unknown_token_Religion",
        "value_JSON_Religion",
        "shared_extracted_Religion",
    ]:
        df.loc[df[col] == "No Affiliation", col] = 0
        df.loc[df[col] == "Christian", col] = 1
    elif col == "revealed_Gender":
        df.loc[df[col] == "male", col] = 0
        df.loc[df[col] == "female", col] = 1
    elif col == "annotator_age":
        df.loc[df[col] == "18-34", col] = 0
        df.loc[df[col] == "46-54", col] = 1
        df.loc[df[col] == "55+", col] = 1
    elif col in ["annotator_gender", "label"]:
        df.loc[df[col] == "male", col] = 0
        df.loc[df[col] == "female", col] = 1
    elif col == "annotator_education_level":
        df.loc[df[col] == "Some or complete graduate degree", col] = 0
        df.loc[df[col] == "(At most) Complete Secondary", col] = 1
        df.loc[df[col] == "Some post-secondary", col] = 1
    elif col == "annotator_political":
        df.loc[df[col] == "Somewhat left-leaning", col] = 0
        df.loc[df[col] == "Very left-leaning", col] = 0
        df.loc[df[col] == "Somewhat right-leaning", col] = 1
        df.loc[df[col] == "Very right-leaning", col] = 1
    elif col == "annotator_ethnicity":
        df.loc[df[col] == "White", col] = 0
        df.loc[df[col] == "Black or African American", col] = 1

    elif col == "age":
        df.loc[df[col] == "18-24 years old", col] = 0
        df.loc[df[col] == "55-64 years old", col] = 1
        df.loc[df[col] == "65+ years old", col] = 1
    elif col == "gender":
        df.loc[df[col] == "Male", col] = 0
        df.loc[df[col] == "Female", col] = 1
    elif col == "gender_nonbinary":
        col = "gender"
        df.loc[df[col] == "Male", col] = 0
        df.loc[df[col] == "Female", col] = 0
        df.loc[df[col] == "Non-binary / third gender", col] = 1
    elif col == "religion":
        df.loc[df[col] == "No Affiliation", col] = 0
        df.loc[df[col] == "Christian", col] = 1
        # df.loc[df[col] == "Jewish", col] = 1
        # df.loc[df[col] == "Muslim", col] = 1
    elif col == "ethnicity":
        df.loc[df[col] == "White", col] = 0
        # df.loc[df[col] == "Hispanic", col] = 1
        df.loc[df[col] == "Black", col] = 1
        # df.loc[df[col] == "Asian", col] = 1
        # df.loc[df[col] == "Mixed", col] = 1
    elif col == "employment_status":
        df.loc[df[col] == "Unemployed, seeking work", col] = 0
        df.loc[df[col] == "Unemployed, not seeking work", col] = 0
        # df.loc[df[col] == "Homemaker / Stay-at-home parent", col] = 0
        df.loc[df[col] == "Working full-time", col] = 1
        df.loc[df[col] == "Working part-time", col] = 1
    elif col == "education":
        df.loc[df[col] == "Some Primary", col] = 0
        df.loc[df[col] == "Completed Primary School", col] = 0
        df.loc[df[col] == "Some Secondary", col] = 0
        df.loc[df[col] == "Completed Secondary School", col] = 0
        df.loc[df[col] == "Graduate / Professional degree", col] = 1
    elif col == "birth_region":
        df.loc[df[col] == "Europe", col] = 0
        df.loc[df[col] == "Americas", col] = 1
    elif col == "reside_region":
        df.loc[df[col] == "Europe", col] = 0
        df.loc[df[col] == "Americas", col] = 1
    elif col == "marital_status":
        df.loc[df[col] == "Never been married", col] = 0
        df.loc[df[col] == "Married", col] = 1
    elif col == "english_proficiency":
        df.loc[df[col] == "Native speaker", col] = 0
        df.loc[df[col] == "Advanced", col] = 1
        df.loc[df[col] == "Intermediate", col] = 1
        df.loc[df[col] == "Basic", col] = 1
    elif col == "lm_familiarity":
        df.loc[df[col] == "Not familiar at all"] = 0
        df.loc[df[col] == "Very familiar"] = 1
    selected_df = df[df[col].isin([0, 1])]
    return selected_df, col


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
    def __init__(self, original_layer, direction, alpha=1.0):
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer

        self.alpha = alpha

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
        ablated = hidden_states - self.alpha * proj

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
        "-item2",
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

    if os.path.isfile(
        f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_dim_{args.domain}_{args.item}{'_'+ args.item2 if args.item2 else ''}_answers.gz"
    ):
        df = pd.read_pickle(
            f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_dim_{args.domain}_{args.item}{'_'+ args.item2 if args.item2 else ''}_answers.gz"
        )
    else:
        df = pd.read_pickle(
            f"{args.data_folder}/{args.dataset}_preprocessed.gz"
        )

    if args.dataset == "prism":
        convo_func = get_prism_convos
        # dim_dataset = "cad_en"
        # dim_convo_func = get_cad_convos

    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        # dim_dataset = "prism"
        # dim_convo_func = get_prism_convos

    dim_df = df
    dim_convo_func = convo_func

    if os.path.isfile(
        f"{args.results_folder}/{args.dataset}_dim_{args.item}{'_'+ args.item2 if args.item2 else ''}.pkl"
    ):
        with open(
            f"{args.results_folder}/{args.dataset}_dim_{args.item}{'_'+ args.item2 if args.item2 else ''}.pkl",
            "rb",
        ) as infile:
            ids_dict = pickle.load(infile)
        in_ids = ids_dict["in_ids"]
        out_ids = ids_dict["out_ids"]

    else:
        dim_linguistic_df = pd.read_pickle(
            f"data/{args.dataset}_utterances_linguistic.gz"
        ).drop(
            columns=["s_neutral_model_response", "s_neutral_user_prompt"],
            errors="ignore",
        )

        # dim_df = pd.read_pickle(
        #     f"{args.data_folder}/{dim_dataset}_preprocessed.gz"
        # )
        # dim_linguistic_df = pd.read_pickle(
        #     f"data/{dim_dataset}_utterances_linguistic.gz"
        # ).drop(
        #     columns=["s_neutral_model_response", "s_neutral_user_prompt"],
        #     errors="ignore",
        # )
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
            dim_linguistic_df,
            args.item,
            args.item2,
        )

        with open(
            f"{args.results_folder}/{args.dataset}_dim_{args.item}{'_'+ args.item2 if args.item2 else ''}.pkl",
            "wb",
        ) as outfile:
            pickle.dump({"in_ids": in_ids, "out_ids": out_ids}, outfile)

    # if args.item2:
    #     in_ids2, out_ids2 = get_example_ids(
    #         dim_linguistic_df,
    #         args.item2,
    #     )

    if args.domain:
        df_questions = pd.read_pickle(
            f"{args.data_folder}/{args.model.split('/')[1]}_questions.gz"
        )
        df_questions = df_questions.loc[df_questions["domain"] == args.domain]
        df_questions = df_questions[
            ~df_questions["q_id"].isin(df.columns.values)
        ].reset_index(drop=True)

        convos = convo_func(df[~df["conversation_id"].isin(in_ids + out_ids)])

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
    end_score = lr.score(np.array(after_0 + after_1)[test_ids], y_test)
    print(beta.norm(p=torch.inf))
    print(
        "end score half",
        end_score,
    )

    # if args.item2:
    #     dim_in_convos = dim_convo_func(
    #         dim_df[dim_df["conversation_id"].isin(in_ids2)]
    #     )
    #     dim_out_convos = dim_convo_func(
    #         dim_df[dim_df["conversation_id"].isin(out_ids2)]
    #     )

    #     inputs_0 = [
    #         tokenizer.apply_chat_template(
    #             convo,
    #             tokenize=True,
    #             add_generation_prompt=True,
    #             return_tensors="pt",
    #             return_dict=False,
    #         )
    #         for convo in dim_in_convos
    #     ]
    #     inputs_1 = [
    #         tokenizer.apply_chat_template(
    #             convo,
    #             tokenize=True,
    #             add_generation_prompt=True,
    #             return_tensors="pt",
    #             return_dict=False,
    #         )
    #         for convo in dim_out_convos
    #     ]

    #     representations_0 = np.array(
    #         [
    #             [
    #                 rep[-1, -1, :].detach().cpu().clone().to(torch.float)
    #                 for rep in model(
    #                     inp.to(device),
    #                     do_sample=False,
    #                     output_hidden_states=True,
    #                     max_new_tokens=1,
    #                     return_dict=True,
    #                 )["hidden_states"]
    #             ]
    #             for inp in tqdm(inputs_0)
    #         ]
    #     )
    #     representations_1 = np.array(
    #         [
    #             [
    #                 rep[-1, -1, :].detach().cpu().clone().to(torch.float)
    #                 for rep in model(
    #                     inp.to(device),
    #                     do_sample=False,
    #                     output_hidden_states=True,
    #                     max_new_tokens=1,
    #                     return_dict=True,
    #                 )["hidden_states"]
    #             ]
    #             for inp in tqdm(inputs_1)
    #         ]
    #     )

    #     layers = get_model_layers(model)
    #     for layer_idx in range(1, args.n_layers):

    #         # Extract hidden states at the specified layer and position
    #         hidden_0 = [torch.tensor(r[layer_idx]) for r in representations_0]
    #         hidden_1 = [torch.tensor(r[layer_idx]) for r in representations_1]

    #         if layer_idx == (args.n_layers - 1):
    #             train_ids, test_ids, y_train, y_test = train_test_split(
    #                 range(len(hidden_0 + hidden_1)),
    #                 [0] * len(hidden_0) + [1] * len(hidden_1),
    #                 test_size=0.33,
    #                 random_state=42,
    #                 stratify=[0] * len(hidden_0) + [1] * len(hidden_1),
    #             )
    #             lr = LogisticRegression(max_iter=1000).fit(
    #                 np.array(hidden_0 + hidden_1)[train_ids],
    #                 y_train,
    #             )
    #             beta = torch.from_numpy(lr.coef_)
    #             print(beta.norm(p=torch.inf))
    #             print(
    #                 "start score half",
    #                 lr.score(
    #                     np.array(hidden_0 + hidden_1)[test_ids],
    #                     y_test,
    #                 ),
    #             )

    #         # Compute mean of hidden states for each category
    #         mean_0 = torch.stack(hidden_0).mean(dim=0)
    #         mean_1 = torch.stack(hidden_1).mean(dim=0)

    #         # Compute refusal direction as the normalized difference between harmful and harmless means
    #         concept_dir = mean_0 - mean_1
    #         concept_dir = concept_dir / concept_dir.norm()
    #         concept_dir = concept_dir.to(device)
    #         model.model.layers[layer_idx] = AblationDecoderLayer(
    #             layers[layer_idx], concept_dir.unsqueeze(dim=0)
    #         )

    #     with torch.no_grad():
    #         after_0 = [
    #             model(
    #                 inp.to(device),
    #                 do_sample=False,
    #                 max_new_tokens=1,
    #             )[
    #                 "logits"
    #             ][-1, -1, :]
    #             .detach()
    #             .cpu()
    #             .clone()
    #             .to(torch.float)
    #             for inp in tqdm(inputs_0)
    #         ]
    #         after_1 = [
    #             model(
    #                 inp.to(device),
    #                 do_sample=False,
    #                 max_new_tokens=1,
    #             )[
    #                 "logits"
    #             ][-1, -1, :]
    #             .detach()
    #             .cpu()
    #             .clone()
    #             .to(torch.float)
    #             for inp in tqdm(inputs_1)
    #         ]

    #     lr = LogisticRegression(max_iter=1000).fit(
    #         np.array(after_0 + after_1)[train_ids], y_train
    #     )
    #     beta = torch.from_numpy(lr.coef_)
    #     print(beta.norm(p=torch.inf))
    #     print(
    #         "end score half",
    #         lr.score(np.array(after_0 + after_1)[test_ids], y_test),
    #     )

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
            tokens = 1 if args.domain != "salary" else 10
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
                f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_dim_{args.domain}_{args.item}{'_'+ args.item2 if args.item2 else ''}_answers.gz"
            )
    elif args.demographic:
        representations = get_representations(
            df, convo_func, tokenizer, model, device
        )
        non_balanced_df = df.loc[
            ~(df[args.demographic].isna())
            & (df[args.demographic] != "Prefer not to say")
            & (df[args.demographic] != "Other")
            & (df[args.demographic] != "Unknown")
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
                + f"/{args.model.split('/')[1]}_{args.dataset}_dim_{args.demographic.replace(' ','')}{specific_label}_{args.item}{'_'+ args.item2 if args.item2 else ''}_mlp_scores.pkl"
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
                + f"/{args.model.split('/')[1]}_{args.dataset}_dim_{args.demographic.replace(' ','')}{specific_label}_{args.item}{'_'+ args.item2 if args.item2 else ''}_mlp_scores.pkl",
                "wb",
            ) as outfile:
                pickle.dump(scores, outfile)
