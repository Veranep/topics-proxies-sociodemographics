import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from huggingface_hub import login

from probing_new import get_model_name

np.random.seed(42)


def get_convo(row):
    return [
        {
            "role": turn["role"].replace("model", "assistant"),
            "content": turn["content"],
        }
        for turn in row["conversation_history"]
        if turn["role"] == "user" or turn["if_chosen"] == True
    ]


def train_probe(df, model, n_layers, demographic, device, prompt, last):
    accuracies = {n: [] for n in range(n_layers)}
    for _ in tqdm(range(5)):
        df_train, df_test = train_test_split(df, shuffle=True)
        train_convos = [get_convo(df.iloc[i]) for i in range(len(df_train))]
        for convo in train_convos:
            to_remove = []
            for i in range(len(convo)):
                if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                    to_remove.append(i)
            to_remove = to_remove[::-1]
            for idx in to_remove:
                del convo[idx]

        if tokenizer.chat_template:
            train_inputs = [
                (
                    tokenizer.encode(
                        tokenizer.apply_chat_template(
                            convo,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        + f" I think the {demographic} of this user is ",
                        return_tensors="pt",
                    )
                    if prompt
                    else tokenizer.apply_chat_template(
                        convo,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                )
                for convo in train_convos
            ]
        else:
            train_inputs = [
                tokenizer(inp, return_tensors="pt") for inp in train_convos
            ]
        if last:
            train_representations = [
                (
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
                    if tokenizer.chat_template
                    else [
                        rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                        for rep in model(
                            **inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                )
                for inp in tqdm(train_inputs)
            ]
        else:
            train_representations = [
                (
                    [
                        torch.mean(
                            rep[-1, :, :]
                            .detach()
                            .cpu()
                            .clone()
                            .to(torch.float),
                            0,
                        )
                        for rep in model(
                            inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                    if tokenizer.chat_template
                    else [
                        torch.mean(
                            rep[-1, :, :]
                            .detach()
                            .cpu()
                            .clone()
                            .to(torch.float),
                            0,
                        )
                        for rep in model(
                            **inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                )
                for inp in tqdm(train_inputs)
            ]
        print("Got train representations")

        test_convos = [get_convo(df.iloc[i]) for i in range(len(df_test))]
        for convo in test_convos:
            to_remove = []
            for i in range(len(convo)):
                if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                    to_remove.append(i)
            to_remove = to_remove[::-1]
            for idx in to_remove:
                del convo[idx]

        if tokenizer.chat_template:
            test_inputs = [
                (
                    tokenizer.encode(
                        tokenizer.apply_chat_template(
                            convo,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                        + f" I think the {demographic} of this user is ",
                        return_tensors="pt",
                    )
                    if prompt
                    else tokenizer.apply_chat_template(
                        convo,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                )
                for convo in test_convos
            ]
        else:
            test_inputs = [
                tokenizer(inp, return_tensors="pt") for inp in test_convos
            ]
        if last:
            test_representations = [
                (
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
                    if tokenizer.chat_template
                    else [
                        rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                        for rep in model(
                            **inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                )
                for inp in tqdm(test_inputs)
            ]
        else:
            test_representations = [
                (
                    [
                        torch.mean(
                            rep[-1, :, :]
                            .detach()
                            .cpu()
                            .clone()
                            .to(torch.float),
                            0,
                        )
                        for rep in model(
                            inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                    if tokenizer.chat_template
                    else [
                        torch.mean(
                            rep[-1, :, :]
                            .detach()
                            .cpu()
                            .clone()
                            .to(torch.float),
                            0,
                        )
                        for rep in model(
                            **inp.to(device),
                            do_sample=False,
                            output_hidden_states=True,
                            max_new_tokens=1,
                            return_dict=True,
                        )["hidden_states"]
                    ]
                )
                for inp in tqdm(test_inputs)
            ]
        print("Got test representations")
        print("Training probe")
        for l in tqdm(range(n_layers)):
            X_train = [rep[l] for rep in train_representations]
            X_test = [rep[l] for rep in test_representations]
            y_train = np.array(df_train["revealed_belief"].tolist())
            y_test = np.array(df_test["revealed_belief"].tolist())
            clf = LogisticRegression(
                random_state=42,
            )
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies[l].append(
                {
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                    "accuracy": accuracy_score(y_test, y_pred),
                }
            )
    return accuracies


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
        "-f",
        "--folder",
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
        "-rb_demo",
        "--revealed_belief_demographic",
        type=str,
        default=None,
        help="Demographic for obtaining revealed belief",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Whether to get representations from the last token position only",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Whether to add an introspective sentence to the prompt",
    )

    args = parser.parse_args()
    if args.token:
        login(args.token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    belief_df = pd.read_pickle(
        f"{args.folder}/{args.model.split('/')[1]}_answers_revealed_belief_judged_{args.revealed_belief_demographic}.gz"
    )

    belief_df["distinct_belief"] = belief_df.groupby(["conversation_id"])[
        "revealed_belief"
    ].transform("nunique")
    belief_df = belief_df.loc[belief_df["distinct_belief"] == 1]
    belief_df = belief_df.groupby(["conversation_id"]).first().reset_index()

    accuracies = train_probe(
        belief_df,
        model,
        args.n_layers,
        args.revealed_belief_demographic,
        device,
        args.prompt,
        args.last,
    )
    with open(
        args.folder
        + f"/{args.model.split('/')[1]}_{args.revealed_belief_demographic}{'_prompt' if args.prompt else ''}{'_last' if args.last else ''}_revealed_belief_results.pkl",
        "wb",
    ) as outfile:
        pickle.dump(accuracies, outfile)
