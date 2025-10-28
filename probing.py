import argparse
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, f1_score

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

chat_templates = {
    "allenai/OLMo-2-1124-7B": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}",
    "allenai/OLMo-2-1124-13B": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}",
}


def balanced_subsample(df, col):
    vals = {
        "age": [],
        "gender": [],
        "religion": ["No Affiliation", "Christian"],
        "ethnicity": ["White"],
        "employment_status": ["Working full-time"],
        "education": ["University Bachelors Degree"],
        "birth_region": ["Europe", "Americas"],
        "reside_region": ["Europe", "Americas"],
        "marital_status": ["Never been married", "Married"],
        "english_proficiency": ["Native speaker", "Fluent"],
    }

    if not len(vals[col]):
        return df

    val_counts = df[col].value_counts()
    for i, key in enumerate(val_counts.keys()):
        if key not in vals[col]:
            max_amount = val_counts[i]
            break
    print(val_counts, max_amount)

    indices_to_drop = []
    for val in vals[col]:
        samples = df[df[col] == val].index.values
        indexes = np.random.choice(samples, size=max_amount, replace=False)
        indices_to_drop += [idx for idx in samples if idx not in indexes]

    return df.drop(index=indices_to_drop)


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

    return selected_df.drop(index=indices_to_drop)


def rebalance(df):
    df["age"] = df["age"].replace(
        {
            "35-44 years old": "35-54 years old",
            "45-54 years old": "35-54 years old",
            "55-64 years old": "55+ years old",
            "65+ years old": "55+ years old",
        }
    )
    df["employment_status"] = df["employment_status"].replace(
        {
            "Unemployed, not seeking work": "Non-Working",
            "Unemployed, seeking work": "Non-Working",
            "Homemaker / Stay-at-home parent": "Non-Working",
            "Retired": "Non-Working",
        }
    )
    df["education"] = df["education"].replace(
        {
            "Some Secondary": "Did Not Complete Secondary School",
            "Completed Primary School": "Did Not Complete Secondary School",
            "Some Primary": "Did Not Complete Secondary School",
        }
    )
    df["marital_status"] = df["marital_status"].replace(
        {
            "Divorced / Separated": "Divorced / Widowed",
            "Widowed": "Divorced / Widowed",
        }
    )
    df["english_proficiency"] = df["english_proficiency"].replace(
        {
            "Advanced": "Non-Fluent",
            "Intermediate": "Non-Fluent",
            "Basic": "Non-Fluent",
        }
    )
    return df


def get_repr(df, dataset, model, tokenizer, device, agg_method, questions):
    if dataset == "prism":
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
                    add_generation_prompt=False,
                    return_tensors="pt",
                )
                for convo in convos
            ]
    elif dataset == "trustpilot":
        inputs = [
            tokenizer(inp, return_tensors="pt") for inp in df["text"].tolist()
        ]
    if agg_method == "last":
        representations = [
            (
                [
                    rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                    for rep in model(
                        inp.to(device),
                        output_hidden_states=True,
                        max_new_tokens=1,
                        return_dict=True,
                    )["hidden_states"]
                ]
                if dataset == "prism"
                else [
                    rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                    for rep in model(
                        **inp.to(device),
                        output_hidden_states=True,
                        max_new_tokens=1,
                        return_dict=True,
                    )["hidden_states"]
                ]
            )
            for inp in tqdm(inputs)
        ]
    elif agg_method == "mean":
        representations = [
            (
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
                if dataset == "prism"
                else [
                    torch.mean(
                        rep[-1, :, :].detach().cpu().clone().to(torch.float), 0
                    )
                    for rep in model(
                        **inp.to(device),
                        output_hidden_states=True,
                        max_new_tokens=1,
                        return_dict=True,
                    )["hidden_states"]
                ]
            )
            for inp in tqdm(inputs)
        ]

    df["representations"] = representations

    return df


def train_probe(
    df,
    dataset,
    n_layers,
    results_file,
    demographic_cols,
    save=False,
    balanced=False,
    rebalance=False,
    twoclasses=False,
    save_file="",
):
    results = {}
    standard_scoring = {
        "f1_micro": "f1_micro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }
    for demographic_col in tqdm(demographic_cols):
        if demographic_col == "age" and dataset == "trustpilot":
            df = df[(df["age"] < 35) | (df["age"] > 45)]
            df["age"] = np.where(df["age"] < 35, "young", "old")
        if rebalance:
            rebalance_df = balanced_subsample(df, demographic_col)
        elif twoclasses:
            twoclasses_df = select_twoclasses(df, demographic_col)
        results[demographic_col] = []
        for l in tqdm(range(n_layers)):
            if dataset == "prism":
                if rebalance:
                    X = np.array(
                        [
                            rep[l]
                            for rep in rebalance_df["representations"].tolist()
                        ]
                    )
                    y = np.array(rebalance_df[demographic_col].tolist())
                elif twoclasses:
                    X = np.array(
                        [
                            rep[l]
                            for rep in twoclasses_df[
                                "representations"
                            ].tolist()
                        ]
                    )
                    y = np.array(twoclasses_df[demographic_col].tolist())
                else:
                    X = np.array(
                        [rep[l] for rep in df["representations"].tolist()]
                    )
                    y = np.array(df[demographic_col].tolist())
                keep_idx = np.where(y != "Prefer not to say")[0]
                y = y[keep_idx]
                X = X[keep_idx]
                if not twoclasses:
                    demo_scoring = {
                        f"f1_{group}": make_scorer(
                            f1_score,
                            average="weighted",
                            labels=[group],
                            pos_label=None,
                        )
                        for group in np.unique(y)
                    }
                    demo_scoring.update(standard_scoring)
                else:
                    demo_scoring = ["f1"]

            elif dataset == "trustpilot":
                X_train = np.array(
                    [
                        rep[l]
                        for rep in df[df["label"] == "train"][
                            "representations"
                        ].tolist()
                    ]
                )
                y_train = np.array(
                    df[df["label"] == "train"][demographic_col]
                    .replace({"F": 1, "M": 0, "young": 1, "old": 0})
                    .tolist()
                )
                X_test = np.array(
                    [
                        rep[l]
                        for rep in df[df["label"] == "test"][
                            "representations"
                        ].tolist()
                    ]
                )
                y_test = np.array(
                    df[df["label"] == "test"][demographic_col]
                    .replace({"F": 1, "M": 0, "young": 1, "old": 0})
                    .tolist()
                )

            clf = LogisticRegression(
                random_state=42, class_weight="balanced" if balanced else None
            )

            if save:
                clf = clf.fit(X, y)
                with open(
                    save_file + f"_{demographic_col}_{l}.pkl", "wb"
                ) as outfile:
                    pickle.dump(clf, outfile)
            else:
                if dataset == "prism":
                    scores = cross_validate(
                        clf,
                        X,
                        y,
                        cv=5,
                        scoring=demo_scoring,
                    )
                    results[demographic_col].append(
                        {
                            metric: scores[f"test_{metric}"]
                            for metric in demo_scoring
                        }
                    )
                elif dataset == "trustpilot":
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    results[demographic_col].append(
                        {"f1": f1_score(y_test, y_pred)}
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
        "-d",
        "--dataset",
        type=str,
        default="prism",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-am",
        "--agg_method",
        type=str,
        default="mean",
        help="Method for aggregating representations across a conversation",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation/"
    )
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        choices=["representations", "probe_cv", "probe_save"],
    )
    parser.add_argument(
        "--add_questions", action="store_true", help="Probe after question"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Whether probe class weight is balanced",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Whether classes should be manually rebalanced",
    )
    parser.add_argument(
        "--twoclasses",
        action="store_true",
        help="Whether probe should only be trained for two large classes",
    )
    args = parser.parse_args()
    if args.mode == "representations":
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
            args.folder
            + f"{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_{args.agg_method}_representations.gz"
        ):
            df = pd.read_pickle(
                args.folder
                + f"{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_{args.agg_method}_representations.gz",
                compression="gzip",
            )
        else:
            if args.dataset == "prism":
                if os.path.isfile(args.folder + "prism_preprocessed.gz"):
                    df = pd.read_pickle(
                        args.folder + "prism_preprocessed.gz",
                        compression="gzip",
                    )
                else:
                    conversations = load_dataset(
                        "HannahRoseKirk/prism-alignment", "conversations"
                    )["train"].to_pandas()
                    survey = load_dataset(
                        "HannahRoseKirk/prism-alignment", "survey"
                    )["train"].to_pandas()

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

                    df.to_pickle(args.folder + "prism_preprocessed.gz")

            elif args.dataset == "trustpilot":
                df = pd.concat(
                    [
                        pd.read_excel("data/en_us_TRAIN.xlsx"),
                        pd.read_excel("data/en_us_TEST.xlsx"),
                    ],
                    ignore_index=True,
                ).drop(columns=["Unnamed: 0", "age_cat"])
                df = df[~pd.isna(df["text"])]
            df = get_repr(
                df,
                args.dataset,
                model,
                tokenizer,
                device,
                args.agg_method,
                questions=questions if args.add_questions else None,
            )
            df.to_pickle(
                args.folder
                + f"{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_{args.agg_method}_representations.gz"
            )

            # questions = list(
            #     set(
            #         pd.read_csv("old/medical_llama_prompts.csv")[
            #             "prompts"
            #         ].tolist()
            #         + pd.read_csv("old/medical_qwen_prompts.csv")[
            #             "prompts"
            #         ].tolist(),
            #     )
            # )

    df = pd.read_pickle(
        args.folder
        + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_{args.agg_method}_representations.gz",
        compression="gzip",
    )

    if args.dataset == "prism":
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
    elif args.dataset == "trustpilot":
        demographic_cols = ["gender", "age"]  # make sure age is second
    n_layers = len(df.iloc[0]["representations"])

    if args.rebalance:
        df = rebalance(df)

    if args.mode == "probe_cv":
        train_probe(
            df,
            args.dataset,
            n_layers,
            args.folder
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_probe_results_{args.agg_method}{'_balanced' if args.balanced else ''}{'_rebalance' if args.rebalance else ''}{'_twoclasses' if args.twoclasses else ''}.pkl",
            demographic_cols,
            save=False,
            balanced=args.balanced,
            rebalance=args.rebalance,
            twoclasses=args.twoclasses,
            save_file=args.folder
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}{'_balanced' if args.balanced else ''}{'_rebalance' if args.rebalance else ''}{'_twoclasses' if args.twoclasses else ''}_probe",
        )

    elif args.mode == "probe_save":
        train_probe(
            df,
            args.dataset,
            n_layers,
            args.folder
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}_probe_results_{args.agg_method}{'_balanced' if args.balanced else ''}{'_rebalance' if args.rebalance else ''}{'_twoclasses' if args.twoclasses else ''}.pkl",
            demographic_cols,
            save=True,
            balanced=args.balanced,
            rebalance=args.rebalance,
            twoclasses=args.twoclasses,
            save_file=args.folder
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset != 'prism' else ''}{'_balanced' if args.balanced else ''}{'_rebalance' if args.rebalance else ''}{'_twoclasses' if args.twoclasses else ''}_probe",
        )
