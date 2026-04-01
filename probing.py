import argparse
import os
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from huggingface_hub import login
from deepsig import aso

from preprocess_data import get_prism_convos, get_cad_convos, get_chen_convos


# np.random.seed(42)


def balance_df(df, col, language):
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
    selected_df = df[df[col].isin([0, 1])]
    max_amount = list(selected_df[col].value_counts())[-1]

    indices_to_drop = []
    for val in [0, 1]:
        samples = selected_df[selected_df[col] == val].index.values
        indexes = np.random.choice(samples, size=max_amount, replace=False)
        indices_to_drop += [idx for idx in samples if idx not in indexes]

    return selected_df.drop(index=indices_to_drop)


def get_representations(df, convo_func, tokenizer, model, device):
    convos = convo_func(df)
    if tokenizer.chat_template:
        inputs = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
            for convo in convos
        ]
    else:
        inputs = [tokenizer(inp, return_tensors="pt") for inp in convos]
    representations = np.array(
        [
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
            for inp in tqdm(inputs)
        ]
    )
    return representations


def train_probe(
    target,
    representations,
    dataset,
    n_layers,
    demographic,
    save,
    save_file,
    mlp,
):
    scores = {
        n: {"f1": [], "majority f1": [], "random f1": []}
        for n in range(n_layers)
    }
    for r in tqdm(range(5)):
        if save and r > 0:
            break
        y_train, y_test, train_representations, test_representations = (
            train_test_split(target, representations, shuffle=True)
        )
        # if dataset == "chen":
        #     train_indices = np.concat(
        #         [train_indices, indices_train + 1, indices_train + 2]
        #     )
        #     df_train = df_train.loc[df_train.index.repeat(3)].reset_index(
        #         drop=True
        #     )
        #     test_indices = np.concat(
        #         [test_indices, indices_test + 1, indices_test + 2]
        #     )
        #     df_test = df_test.loc[df_test.index.repeat(3)].reset_index(
        #         drop=True
        #     )
        print("Training probe")
        values, counts = np.unique(y_test, return_counts=True)
        majority = values[np.argmax(counts)]
        majority_f1 = f1_score(
            y_test,
            np.full(len(y_test), majority),
            average="macro",
        )
        random_f1 = f1_score(
            y_test,
            np.random.choice(
                np.unique(y_test), size=len(y_test), replace=True
            ),
            average="macro",
        )
        for l in tqdm(range(n_layers)):
            X_train = [rep[l] for rep in train_representations]
            X_test = [rep[l] for rep in test_representations]
            if mlp:
                clf = MLPClassifier(random_state=42)
            else:
                clf = LogisticRegression(random_state=42)
            clf = clf.fit(X_train, y_train)
            # if save:
            #     with open(
            #         save_file + f"_{demographic}_{l}.pkl", "wb"
            #     ) as outfile:
            #         pickle.dump(clf, outfile)
            # else:
            y_pred = clf.predict(X_test)
            scores[l]["f1"].append(f1_score(y_test, y_pred, average="macro"))
            scores[l]["majority f1"].append(majority_f1)
            scores[l]["random f1"].append(random_f1)
    for l in range(n_layers):
        scores[l]["majority aso"] = aso(
            scores[l]["f1"], scores[l]["majority f1"], seed=42
        )
        scores[l]["random aso"] = aso(
            scores[l]["f1"], scores[l]["random f1"], seed=42
        )
    # if save:
    #     id_col = "conversation_id" if dataset != "chen" else "text_id"
    #     with open(save_file + f"_{demographic}_ids.pkl", "wb") as outfile:
    #         pickle.dump(
    #             (
    #                 df_train[id_col].to_numpy(),
    #                 df_test[id_col].to_numpy(),
    #             ),
    #             outfile,
    #         )
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
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Whether to an MLP probe instead of a linear probe",
    )
    args = parser.parse_args()
    if "belief" in args.data_folder:
        df = pd.read_pickle(
            f"{args.data_folder}/Llama-3.1-8B-Instruct_{args.dataset}_beliefs_preprocessed.gz"
        )
    else:
        df = pd.read_pickle(
            f"{args.data_folder}/{args.dataset}_preprocessed.gz"
        )

    if os.path.isfile(
        args.results_folder
        + f"/{args.model.split('/')[1]}_{args.dataset}_{args.demographic.replace(' ','')}{'_balanced' if args.balanced else ''}{'_mlp' if args.mlp else ''}_scores.pkl"
    ):
        pass

    elif os.path.isfile(
        args.results_folder
        + f"/{args.model.split('/')[1]}_{args.dataset}_representations.pkl"
    ):
        with open(
            args.results_folder
            + f"/{args.model.split('/')[1]}_{args.dataset}_representations.pkl",
            "rb",
        ) as infile:
            representations = pickle.load(infile)

        if args.balanced:
            df = balance_df(
                df,
                args.demographic,
                args.dataset.split("_")[-1] if "cad" in args.dataset else "",
            )
        else:
            df = df.loc[
                ~(df[args.demographic].isna())
                & (df[args.demographic] != "Prefer not to say")
                & (df[args.demographic] != "Other")
                & (df[args.demographic] != "Unknown")
                & (df[args.demographic] != "female-male-non-binary")
            ]

        print(df.index, len(df.index), len(representations))
        representations = representations[df.index]
        print(len(representations))

        print("ready df")

        scores = train_probe(
            df[args.demographic].tolist(),
            representations,
            args.dataset,
            args.n_layers,
            args.demographic,
            save=args.save,
            save_file=args.results_folder + f"/{args.model.split('/')[1]}",
            mlp=args.mlp,
        )
        with open(
            args.results_folder
            + f"/{args.model.split('/')[1]}_{args.dataset}_{args.demographic.replace(' ','')}{'_balanced' if args.balanced else ''}{'_mlp' if args.mlp else ''}_scores.pkl",
            "wb",
        ) as outfile:
            pickle.dump(scores, outfile)

    else:
        if args.token:
            login(args.token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if args.dataset == "prism":
            convo_func = get_prism_convos
        elif "cad" in args.dataset:
            convo_func = get_cad_convos
        elif args.dataset == "chen":
            convo_func = get_chen_convos
        representations = get_representations(
            df, convo_func, tokenizer, model, device
        )
        with open(
            args.results_folder
            + f"/{args.model.split('/')[1]}_{args.dataset}_representations.pkl",
            "wb",
        ) as outfile:
            pickle.dump(representations, outfile)
        print("got representations!")
