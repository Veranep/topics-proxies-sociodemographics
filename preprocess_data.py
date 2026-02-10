import argparse
from datasets import load_dataset
import pandas as pd
import pickle


def get_prism_turns(row):
    return [
        {
            "role": turn["role"].replace("model", "assistant"),
            "content": turn["content"],
        }
        for turn in row["conversation_history"]
        if turn["role"] == "user" or turn["if_chosen"] == True
    ]


def get_prism_convos(df):
    convos = list(map(get_prism_turns, df.itertuples(index=False)))
    for convo in convos:
        to_remove = []
        for i in range(len(convo)):
            if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for idx in to_remove:
            del convo[idx]
    return convos


def get_cad_turns(row):
    turns = []
    for turn in ["first", "second", "third", "fourth"]:
        pref_response = f"{turn}_turn_preferred_response"
        if not row[pref_response]:
            return turns
        else:
            turns += [
                {"role": "user", "content": row[f"{turn}_turn_prompt"]},
                {
                    "role": "assistant",
                    "content": row[
                        f"{turn}_turn_response_{row[pref_response][-1]}"
                    ],
                },
            ]
    return turns


def get_cad_convos(df):
    return list(map(get_cad_turns, df.itertuples(index=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="prism",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Folder dataset is stored in",
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    if args.dataset == "prism":
        conversations = load_dataset(
            "HannahRoseKirk/prism-alignment", "conversations"
        )["train"].to_pandas()
        survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")[
            "train"
        ].to_pandas()

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
                    dict(x)[region] if type(x) == dict else "Prefer not to say"
                )
            )

        df.to_pickle(f"{args.folder}/prism_preprocessed.gz")
        print(get_prism_convos(df)[:5])
    elif args.dataset == "cad":
        ds = load_dataset(
            "facebook/community-alignment-dataset", split="filtered"
        )
        ds.to_pandas().to_pickle(f"{args.folder}/cad_preprocessed.gz")
        print(get_cad_convos(ds.to_pandas())[:5])
        for language in ["en", "fr", "it", "pt", "hi"]:
            filtered_ds = ds.filter(
                lambda example: example["assigned_lang"] == language
            )
            filtered_ds.to_pandas().to_pickle(
                f"{args.folder}/cad_{language + '_'}preprocessed.gz"
            )
