import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from huggingface_hub import login

np.random.seed(42)


def get_convo(row):
    return [
        {
            "role": turn["role"].replace("model", "assistant"),
            "content": turn["content"],
        }
        for turn in row["conversation_history"]
        if turn["role"] == "user" or turn["if_chosen"] == True
    ] + [{"role": "user", "content": row["question"]}]


def change_labels(df, col):
    if col == "age":
        df.loc[df[col] == "18-24 years old", col] = 0
        df.loc[df[col] == "55-64 years old", col] = 1
        df.loc[df[col] == "65+ years old", col] = 1
    elif col == "gender":
        df.loc[df[col] == "Male", col] = 0
        df.loc[df[col] == "Female", col] = 1
    elif col == "religion":
        df.loc[df[col] == "No Affiliation", col] = 0
        df.loc[df[col] == "Christian", col] = 1
        df.loc[df[col] == "Jewish", col] = 1
        df.loc[df[col] == "Muslim", col] = 1
    elif col == "ethnicity":
        df.loc[df[col] == "White", col] = 0
        df.loc[df[col] == "Hispanic", col] = 1
        df.loc[df[col] == "Black", col] = 1
        df.loc[df[col] == "Asian", col] = 1
        df.loc[df[col] == "Mixed", col] = 1
    elif col == "employment_status":
        df.loc[df[col] == "Unemployed, seeking work", col] = 0
        df.loc[df[col] == "Unemployed, not seeking work", col] = 0
        df.loc[df[col] == "Homemaker / Stay-at-home parent", col] = 0
        df.loc[df[col] == "Working full-time", col] = 1
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
    return df


def select_twoclasses(df, col):
    if col == "age":
        df.loc[df[col] == "18-24 years old", col] = 0
        df.loc[df[col] == "55-64 years old", col] = 1
        df.loc[df[col] == "65+ years old", col] = 1
    elif col == "gender":
        df.loc[df[col] == "Male", col] = 0
        df.loc[df[col] == "Female", col] = 1
    elif col == "religion":
        df.loc[df[col] == "No Affiliation", col] = 0
        df.loc[df[col] == "Christian", col] = 1
        df.loc[df[col] == "Jewish", col] = 1
        df.loc[df[col] == "Muslim", col] = 1
    elif col == "ethnicity":
        df.loc[df[col] == "White", col] = 0
        df.loc[df[col] == "Hispanic", col] = 1
        df.loc[df[col] == "Black", col] = 1
        df.loc[df[col] == "Asian", col] = 1
        df.loc[df[col] == "Mixed", col] = 1
    elif col == "employment_status":
        df.loc[df[col] == "Unemployed, seeking work", col] = 0
        df.loc[df[col] == "Unemployed, not seeking work", col] = 0
        df.loc[df[col] == "Homemaker / Stay-at-home parent", col] = 0
        df.loc[df[col] == "Working full-time", col] = 1
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
    selected_df = df[df[col].isin([0, 1])].reset_index(drop=True)
    max_amount = list(selected_df[col].value_counts())[-1]

    indices_to_drop = []
    for val in [0, 1]:
        samples = selected_df[selected_df[col] == val].index.values
        indexes = np.random.choice(samples, size=max_amount, replace=False)
        indices_to_drop += [idx for idx in samples if idx not in indexes]

    return selected_df.drop(index=indices_to_drop)["conversation_id"].unique()


def train_probe(df, model, n_layers, demographic, device, prompt, last):
    accuracies = {n: [] for n in range(n_layers)}
    select_df = df[df["question"] == df["question"].unique()[0]]
    selected_ids = select_twoclasses(select_df, demographic)
    for _ in tqdm(range(5)):
        train_ids, test_ids = train_test_split(selected_ids, shuffle=True)
        df_train = (
            df[df["conversation_id"].isin(train_ids)]
            .groupby("conversation_id")
            .sample(n=1, random_state=42)
        )
        df_train = change_labels(df_train, demographic)
        df_test = (
            df[df["conversation_id"].isin(test_ids)]
            .groupby("conversation_id")
            .sample(n=1, random_state=42)
        )
        df_test = change_labels(df_test, demographic)
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
            y_train = np.array(df_train[demographic].tolist())
            y_test = np.array(df_test[demographic].tolist())
            clf = LogisticRegression(
                random_state=42,
            )
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracies[l].append({"f1": f1_score(y_test, y_pred)})
    return accuracies


def get_model_name(model):
    if "Olmo-3" in model:
        return "OLMo3"
    elif "OLMo-2" in model:
        return "OLMo2"
    elif "Llama" in model:
        return "Llama"
    elif "gemma" in model:
        return "Gemma"


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
        "-demo",
        "--demographic",
        type=str,
        default=None,
        help="Demographic to train and evaluate probe for",
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
    if not tokenizer.chat_template and not args.first:
        tokenizer.chat_template = chat_templates[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    climate_fever = pd.read_pickle(
        "data/prism_questions_climate_fever.gz",
        compression="gzip",
    )
    health_misinfo = pd.read_pickle(
        "data/prism_questions_health_misinfo.gz",
        compression="gzip",
    )
    pubhealth = pd.read_pickle(
        "data/prism_questions_pubhealth.gz",
        compression="gzip",
    )

    df = pd.concat([climate_fever, health_misinfo, pubhealth])

    with open("data/conv_ids_prism.pkl", "rb") as infile:
        conv_ids = pickle.load(infile)
    with open("data/q_ids_prism.pkl", "rb") as infile:
        q_ids = pickle.load(infile)

    df = df[df["conversation_id"].isin(conv_ids[model_name][args.demographic])]
    df = df[df["question"].isin(q_ids[model_name][args.demographic])]

    accuracies = train_probe(
        df,
        model,
        args.n_layers,
        args.demographic,
        device,
        args.prompt,
        args.last,
    )
    with open(
        args.folder
        + f"/{args.model.split('/')[1]}_{args.demographic}{'_prompt' if args.prompt else ''}{'_last' if args.last else ''}_trainq_results.pkl",
        "wb",
    ) as outfile:
        pickle.dump(accuracies, outfile)
