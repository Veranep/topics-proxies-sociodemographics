import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
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
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in tqdm(
        skf.split(df, df["gender_labels"].astype("int").values)
    ):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        print(df_train.shape, df_test.shape)
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
            y_train = np.array(df_train["gender_labels"].tolist())
            y_test = np.array(df_test["gender_labels"].tolist())
            clf = LogisticRegression(
                random_state=42,
            )
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies[l].append(
                {
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "rocauc": roc_auc_score(
                        y_test, clf.decision_function(X_test)
                    ),
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
    health_misinfo = pd.read_pickle(
        "data/prism_questions_health_misinfo.gz",
        compression="gzip",
    )
    health_misinfo = (
        health_misinfo.groupby(["conversation_id"]).first().reset_index()
    )

    health_misinfo = health_misinfo[health_misinfo["conversation_id"] != ""]

    with open(
        args.folder
        + f"/{args.model.split('/')[1]}_31_prompt_question_kmeans2_results.pkl",
        "rb",
    ) as infile:
        k_means = pickle.load(infile)[2]

    cid_to_label = {"": pd.NA}

    for i in range(len(k_means)):
        cid_to_label[f"c{i}"] = k_means[i]

    health_misinfo["gender_labels"] = health_misinfo["conversation_id"].map(
        cid_to_label
    )

    accuracies = train_probe(
        health_misinfo,
        model,
        args.n_layers,
        "gender",
        device,
        args.prompt,
        args.last,
    )
    with open(
        args.folder
        + f"/{args.model.split('/')[1]}_{'_prompt' if args.prompt else ''}{'_last' if args.last else ''}_gender_kmeans_results.pkl",
        "wb",
    ) as outfile:
        pickle.dump(accuracies, outfile)
