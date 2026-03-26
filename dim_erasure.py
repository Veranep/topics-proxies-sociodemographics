import argparse
from data_selection_erasure import (
    leace_convos_prism,
    leace_convos_cad,
    evaluation_prism,
    evaluation_cad,
)
import datasets
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
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

        print("r shape", self.r.shape)

        # take the unit
        self.r_unit = self.r / torch.norm(self.r)

        print("r unit shape", self.r_unit.shape)

        print("r unit T shape", self.r_unit.T.shape)

        self.projection = torch.matmul(self.r_unit, self.r_unit.T)

        print("projection shape", self.projection.shape)

    def forward(self, *args, **kwargs):
        # get the hidden states
        hidden_states = args[0]

        # apply the projection to all the hidden states
        proj = torch.matmul(hidden_states, self.projection)

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

    labels = list(leace_convos[args.item].keys())

    inputs_0 = []
    inputs_1 = []
    for i, convo in enumerate(leace_cs):
        label = leace_labels.iloc[i]
        if label == labels[0]:
            inputs_0.append(
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                )
            )
        else:
            inputs_1.append(
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                )
            )

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
    for layer_idx in range(args.n_layers):

        # Extract hidden states at the specified layer and position
        hidden_0 = [torch.tensor(r[layer_idx]) for r in representations_0]
        hidden_1 = [torch.tensor(r[layer_idx]) for r in representations_1]

        if layer_idx == (args.n_layers - 1):
            lr = LogisticRegression(max_iter=1000).fit(
                hidden_0 + hidden_1, [0] * len(hidden_0) + [1] * len(hidden_1)
            )
            beta = torch.from_numpy(lr.coef_)
            print(beta.norm(p=torch.inf))
            print(
                lr.score(
                    hidden_0 + hidden_1,
                    [0] * len(hidden_0) + [1] * len(hidden_1),
                )
            )

        # Compute mean of hidden states for each category
        mean_0 = torch.stack(hidden_0).mean(dim=0)
        mean_1 = torch.stack(hidden_1).mean(dim=0)

        # Compute refusal direction as the normalized difference between harmful and harmless means
        concept_dir = mean_0 - mean_1
        concept_dir = concept_dir / concept_dir.norm()
        concept_dir = concept_dir.to(device).to(torch.float16)
        print("shape", concept_dir.shape)
        model.model.layers[layer_idx] = AblationDecoderLayer(
            layers[layer_idx], concept_dir.unsqueeze(dim=0)
        )

    after_0 = [
        model(
            inp.to(device),
            do_sample=False,
            max_new_tokens=1,
        )[-1, -1, :]
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
        )[-1, -1, :]
        .detach()
        .cpu()
        .clone()
        .to(torch.float)
        for inp in tqdm(inputs_1)
    ]

    lr = LogisticRegression(max_iter=1000).fit(
        after_0 + after_1, [0] * len(after_0) + [1] * len(after_1)
    )
    beta = torch.from_numpy(lr.coef_)
    print(beta.norm(p=torch.inf))
    print(
        lr.score(
            after_0 + after_1,
            [0] * len(after_0) + [1] * len(after_1),
        )
    )

    # with scrubber.scrub(model):
    #     if args.domain:
    #         leace_pipeline = pipeline(
    #             "text-generation",
    #             model=model,
    #             tokenizer=tokenizer,
    #             torch_dtype=torch.bfloat16,
    #             device_map="auto",
    #         )
    #         for row in tqdm(df_questions.itertuples(index=False)):
    #             convos_and_questions = [
    #                 tokenizer.apply_chat_template(
    #                     convo + [{"role": "user", "content": row.question}],
    #                     tokenize=False,
    #                     add_generation_prompt=True,
    #                 )
    #                 for convo in convos
    #             ]
    #             tokens = 1
    #             outputs = [
    #                 answer[0]["generated_text"]
    #                 for answer in tqdm(
    #                     leace_pipeline(
    #                         ListDataset(convos_and_questions),
    #                         batch_size=32,
    #                         max_new_tokens=tokens,
    #                         return_full_text=False,
    #                         do_sample=False,
    #                     )
    #                 )
    #             ]

    #             evaluation_df = pd.concat(
    #                 [evaluation_df, pd.DataFrame({row.q_id: outputs})],
    #                 axis=1,
    #             )
    #             evaluation_df.to_pickle(
    #                 f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_leace_{args.domain}_{args.item}_answers.gz"
    #             )
    #     elif args.demographic:
    #         representations = get_representations(
    #             df, convo_func, tokenizer, model, device
    #         )
    #         if args.dataset == "prism":
    #             if args.balanced:
    #                 df = balance_df(df, args.demographic, "")
    #             else:
    #                 df = df.loc[
    #                     ~(df[args.demographic].isna())
    #                     & (df[args.demographic] != "Prefer not to say")
    #                     & (df[args.demographic] != "Unknown")
    #                     & (df[args.demographic] != "female-male-non-binary")
    #                 ]
    #         elif "cad" in args.dataset:
    #             if args.balanced:
    #                 df = balance_df(
    #                     df, args.demographic, args.dataset.split("_")[-1]
    #                 )
    #             else:
    #                 df = df.loc[
    #                     ~(df[args.demographic].isna())
    #                     & (df[args.demographic] != "Prefer not to say")
    #                     & (df[args.demographic] != "other")
    #                     & (df[args.demographic] != "Unknown")
    #                     & (df[args.demographic] != "female-male-non-binary")
    #                 ]
    #         elif args.dataset == "chen":
    #             if args.balanced:
    #                 df = balance_df(df, args.demographic, "")
    #             else:
    #                 df = df.loc[
    #                     ~(df[args.demographic].isna())
    #                     & (df[args.demographic] != "Unknown")
    #                     & (df[args.demographic] != "female-male-non-binary")
    #                 ]
    #         representations = representations[df.index]
    #         scores = train_probe(
    #             df[args.demographic].tolist(),
    #             representations,
    #             args.dataset,
    #             args.n_layers,
    #             args.demographic,
    #             save=args.save,
    #             save_file=args.results_folder + f"/{args.model.split('/')[1]}",
    #         )
    #         with open(
    #             args.results_folder
    #             + f"/{args.model.split('/')[1]}_{args.dataset}_leace_{args.demographic.replace(' ','')}{'_balanced' if args.balanced else ''}_{args.item}_scores.pkl",
    #             "wb",
    #         ) as outfile:
    #             pickle.dump(scores, outfile)
