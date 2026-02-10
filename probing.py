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
from deepsig import aso

from preprocess_data import get_prism_convos, get_cad_convos


np.random.seed(42)


def balance_df(df, col, language):
    if col == "annotator_age":
        df.loc[df[col] == "18-34", col] = 0
        df.loc[df[col] == "46-54", col] = 1
        df.loc[df[col] == "55+", col] = 1
    elif col == "annotator_gender":
        df.loc[df[col] == "male", col] = 0
        df.loc[df[col] == "female", col] = 1
    elif col == "annotator_education_level":
        df.loc[df[col] == "Some or complete graduate degree"] = 0
        df.loc[df[col] == "(At most) Complete Secondary"] = 1
        df.loc[df[col] == "Some post-secondary"] = 1
    elif col == "annotator_political":
        df.loc[df[col] == "Somewhat left-leaning"] = 0
        df.loc[df[col] == "Very left-leaning"] = 0
        df.loc[df[col] == "Somewhat right-leaning"] = 1
        df.loc[df[col] == "Very right-leaning"] = 1
    elif col == "annotator_ethnicity":
        if language == "en":
            df.loc[df[col] == "White"] = 0
            df.loc[df[col] == "Black or African American"] = 1
        elif language == "fr":
            df.loc[df[col] == "Non immigrant"] = 0
            df.loc[df[col] == "Immigrant"] = 1
        elif language == "it":
            df.loc[df[col] == "Italian"] = 0
            df.loc[df[col] == "Foreign national"] = 1
        elif language == "hi":
            df.loc[df[col] == "Indo-Aryan"] = 0
            df.loc[df[col] == "Dravidian"] = 1
        elif language == "pt":
            df.loc[df[col] == "White"] = 0
            df.loc[df[col] == "Brown/Mixed"] = 1
    elif col == "age":
        df.loc[df[col] == "18-24 years old", col] = 0
        df.loc[df[col] == "55-64 years old", col] = 1
        df.loc[df[col] == "65+ years old", col] = 1
    elif col == "gender":
        df.loc[df[col] == "Male", col] = 0
        df.loc[df[col] == "Female", col] = 1
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
    elif col == "lm_familiarity":
        df.loc[df[col] == "Not familiar at all"] = 0
        df.loc[df[col] == "Very familiar"] = 1
    selected_df = df[df[col].isin([0, 1])].reset_index(drop=True)
    max_amount = list(selected_df[col].value_counts())[-1]

    indices_to_drop = []
    for val in [0, 1]:
        samples = selected_df[selected_df[col] == val].index.values
        indexes = np.random.choice(samples, size=max_amount, replace=False)
        indices_to_drop += [idx for idx in samples if idx not in indexes]

    return selected_df.drop(index=indices_to_drop)


def train_probe(
    df, convo_func, model, n_layers, demographic, device, save, save_file
):
    scores = {n: {"f1": [], "majority f1": []} for n in range(n_layers)}
    for r in tqdm(range(5)):
        if save and r > 0:
            break
        df_train, df_test = train_test_split(df, shuffle=True)
        train_convos = convo_func(df_train)

        if tokenizer.chat_template:
            train_inputs = [
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                for convo in train_convos
            ]
        else:
            train_inputs = [
                tokenizer(inp, return_tensors="pt") for inp in train_convos
            ]
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
        print("Got train representations")

        test_convos = convo_func(df_test)

        if tokenizer.chat_template:
            test_inputs = [
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                for convo in test_convos
            ]
        else:
            test_inputs = [
                tokenizer(inp, return_tensors="pt") for inp in test_convos
            ]
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
        print("Got test representations")
        print("Training probe")
        values, counts = np.unique(y_test, return_counts=True)
        majority = values[np.argmax(counts)][0]
        majority_f1 = f1_score(
            y_test,
            np.full(len(y_test), majority),
            average="macro",
        )
        for l in tqdm(range(n_layers)):
            X_train = [rep[l] for rep in train_representations]
            X_test = [rep[l] for rep in test_representations]
            y_train = np.array(df_train[demographic].tolist())
            y_test = np.array(df_test[demographic].tolist())
            clf = LogisticRegression(
                random_state=42,
            )
            clf = clf.fit(X_train, y_train)
            if save:
                with open(
                    save_file + f"_{demographic}_{l}.pkl", "wb"
                ) as outfile:
                    pickle.dump(clf, outfile)
            else:
                y_pred = clf.predict(X_test)
                scores[l]["f1"].append(
                    f1_score(y_test, y_pred, average="macro")
                )
                scores[l]["majority f1"].append(majority_f1)
    for l in range(n_layers):
        scores[l]["aso"] = aso(
            scores[l]["f1"], scores[l]["majority f1"], seed=42
        )
    if save:
        with open(save_file + f"_{demographic}_ids.pkl", "wb") as outfile:
            pickle.dump(
                (
                    df_train["conversation_id"].to_numpy(),
                    df_test["conversation_id"].to_numpy(),
                ),
                outfile,
            )
    return scores


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
        "-demo",
        "--demographic",
        type=str,
        default=None,
        help="Demographic to train probe for",
    )
    parser.add_argument(
        "-n",
        "--n_layers",
        type=int,
        default=None,
        help="Number of model layers",
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
        "--save",
        action="store_true",
        help="Whether to save the trained probe",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Whether to balance the dataset for the demographic attribute",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    df.read_pickle(f"{args.data_folder}/{args.dataset}_preprocessed.gz")
    if args.dataset == "prism":
        if args.balanced:
            df = balance_df(df, args.demographic, "")
        else:
            df = df.loc[
                ~(df[args.demographic].isna()) & df[args.demographic]
                != "Prefer not to say"
            ]
        convo_func = get_prism_convos
    elif "cad" in args.dataset:
        if args.balanced:
            df = balance_df(df, args.demographic, args.dataset.split("_")[-1])
        else:
            df = df.loc[
                ~(df[args.demographic].isna()) & df[args.demographic]
                != "Prefer not to say" & df[args.demographic]
                != "other"
            ]
        convo_func = get_cad_convos

    scores = train_probe(
        convos,
        convo_func,
        model,
        args.n_layers,
        args.demographic,
        device,
        save=args.save,
        save_file=args.results_folder + f"/{args.model.split('/')[1]}",
    )
    with open(
        args.results_folder
        + f"/{args.model.split('/')[1]}_{args.dataset}_{args.demographic}{'_balanced' if args.balanced else ''}_scores.pkl",
        "wb",
    ) as outfile:
        pickle.dump(scores, outfile)
