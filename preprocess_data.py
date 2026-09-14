import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
import numpy as np
import re
import pandas as pd
import pickle
from tqdm import tqdm
from collections import Counter


def get_prism_turns(row, specify_text):
    if specify_text:
        turns = [
            {
                "role": turn["role"].replace("model", "assistant"),
                "content": [{"type": "text", "text": turn["content"]}],
            }
            for turn in row.conversation_history
            if turn["role"] == "user" or turn["if_chosen"] == True
        ]
    else:
        turns = [
            {
                "role": turn["role"].replace("model", "assistant"),
                "content": turn["content"],
            }
            for turn in row.conversation_history
            if turn["role"] == "user" or turn["if_chosen"] == True
        ]
    return turns


def get_prism_convos(df, specify_text=False):
    convos = list(
        map(
            lambda x: get_prism_turns(x, specify_text),
            df.itertuples(index=False),
        )
    )
    for convo in convos:
        to_remove = []
        for i in range(len(convo)):
            if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for idx in to_remove:
            del convo[idx]
    return convos


def get_cad_turns(row, specify_text):
    turns = []
    for turn in ["first", "second", "third", "fourth"]:
        pref_response = f"{turn}_turn_preferred_response"
        if not getattr(row, pref_response):
            return turns
        else:
            if specify_text:
                turns += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": getattr(row, f"{turn}_turn_prompt"),
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": getattr(
                                    row,
                                    f"{turn}_turn_response_{getattr(row,pref_response)[-1]}",
                                ),
                            }
                        ],
                    },
                ]
            else:
                turns += [
                    {
                        "role": "user",
                        "content": getattr(row, f"{turn}_turn_prompt"),
                    },
                    {
                        "role": "assistant",
                        "content": getattr(
                            row,
                            f"{turn}_turn_response_{getattr(row,pref_response)[-1]}",
                        ),
                    },
                ]
    return turns


def get_cad_convos(df, specify_text=False):
    return list(
        map(
            lambda x: get_cad_turns(x, specify_text),
            df.itertuples(index=False),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="prism",
        help="Dataset to preprocess",
    )
    args = parser.parse_args()
    if args.dataset == "prism":
        conversations = load_dataset(
            "HannahRoseKirk/prism-alignment", "conversations"
        )["train"].to_pandas()
        utterances = load_dataset(
            "HannahRoseKirk/prism-alignment", "utterances"
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

        if not os.path.isfile(f"data/prism_preprocessed.gz"):
            df.to_pickle(f"data/prism_preprocessed.gz")
        print(get_prism_convos(df)[:5])

        df_utterances = pd.merge(utterances, survey, on=["user_id"])
        to_simplify = ["religion", "ethnicity"]
        for column in to_simplify:
            df_utterances[column] = df_utterances[column].apply(
                lambda x: (
                    dict(x)["simplified"]
                    if type(x) == dict
                    else "Prefer not to say"
                )
            )
        regions = ["birth_region", "reside_region"]
        for region in regions:
            df_utterances[region] = df_utterances["location"].apply(
                lambda x: (
                    dict(x)[region] if type(x) == dict else "Prefer not to say"
                )
            )
        df_utterances = (
            df_utterances[df_utterances["if_chosen"] == True]
            .groupby(["conversation_id", "turn"])
            .first()
            .reset_index()
        )
        if not os.path.isfile(f"data/prism_utterances_preprocessed.gz"):
            df_utterances.to_pickle(f"data/prism_utterances_preprocessed.gz")

    elif args.dataset == "cad":
        ds = load_dataset(
            "facebook/community-alignment-dataset", split="filtered"
        ).filter(lambda example: example["assigned_lang"] == "en")

        df = ds.to_pandas()
        df["first_turn_preferred_response"] = df.to_numpy()[
            np.arange(len(df)),
            df.columns.get_indexer(
                "first_turn_response_"
                + df["first_turn_preferred_response"].str[-1]
            ),
        ]
        df["second_turn_preferred_response"] = df.to_numpy()[
            np.arange(len(df)),
            df.columns.get_indexer(
                "second_turn_response_"
                + df["second_turn_preferred_response"].str[-1]
            ),
        ]
        df["third_turn_preferred_response"] = df.to_numpy()[
            np.arange(len(df)),
            df.columns.get_indexer(
                "third_turn_response_"
                + df["third_turn_preferred_response"].str[-1]
            ),
        ]
        df["fourth_turn_preferred_response"] = df.to_numpy()[
            np.arange(len(df)),
            df.columns.get_indexer(
                "fourth_turn_response_"
                + df["fourth_turn_preferred_response"].str[-1]
            ),
        ]
        df = df.drop(
            columns=[
                "first_turn_response_a",
                "first_turn_response_b",
                "first_turn_response_c",
                "first_turn_response_d",
                "second_turn_response_a",
                "second_turn_response_b",
                "second_turn_response_c",
                "second_turn_response_d",
                "third_turn_response_a",
                "third_turn_response_b",
                "third_turn_response_c",
                "third_turn_response_d",
                "fourth_turn_response_a",
                "fourth_turn_response_b",
                "fourth_turn_response_c",
                "fourth_turn_response_d",
                "first_turn_responses",
                "second_turn_responses",
                "third_turn_responses",
                "fourth_turn_responses",
            ]
        ).rename(
            columns={
                "first_turn_prompt": "first",
                "second_turn_prompt": "second",
                "third_turn_prompt": "third",
                "fourth_turn_prompt": "fourth",
            }
        )
        df["first_turn_prompt"] = df["first"]
        df = df.melt(
            id_vars=[
                "conversation_id",
                "assigned_lang",
                "annotator_id",
                "first_turn_preferred_response",
                "second_turn_preferred_response",
                "third_turn_preferred_response",
                "fourth_turn_preferred_response",
                "first_turn_feedback",
                "second_turn_feedback",
                "third_turn_feedback",
                "fourth_turn_feedback",
                "annotator_age",
                "annotator_gender",
                "annotator_education_level",
                "annotator_political",
                "annotator_ethnicity",
                "annotator_country",
                "is_pregenerated_first_prompt",
                "in_balanced_subset",
                "in_balanced_subset_10",
                "first_turn_prompt",
            ],
            var_name="turn",
            value_name="user_prompt",
        )
        df = df[~df["user_prompt"].isna()].rename(
            columns={
                "first_turn_preferred_response": "first",
                "second_turn_preferred_response": "second",
                "third_turn_preferred_response": "third",
                "fourth_turn_preferred_response": "fourth",
            }
        )
        df = df.melt(
            id_vars=[
                "conversation_id",
                "assigned_lang",
                "annotator_id",
                "first_turn_feedback",
                "second_turn_feedback",
                "third_turn_feedback",
                "fourth_turn_feedback",
                "annotator_age",
                "annotator_gender",
                "annotator_education_level",
                "annotator_political",
                "annotator_ethnicity",
                "annotator_country",
                "is_pregenerated_first_prompt",
                "in_balanced_subset",
                "in_balanced_subset_10",
                "first_turn_prompt",
                "user_prompt",
                "turn",
            ],
            var_name="turn2",
            value_name="model_response",
        )
        df = (
            df[df["turn"] == df["turn2"]]
            .reset_index(drop=True)
            .drop(columns=["turn2"])
        )
        df = df.rename(
            columns={
                "first_turn_feedback": "first",
                "second_turn_feedback": "second",
                "third_turn_feedback": "third",
                "fourth_turn_feedback": "fourth",
            }
        )
        df = df.melt(
            id_vars=[
                "conversation_id",
                "assigned_lang",
                "annotator_id",
                "annotator_age",
                "annotator_gender",
                "annotator_education_level",
                "annotator_political",
                "annotator_ethnicity",
                "annotator_country",
                "is_pregenerated_first_prompt",
                "in_balanced_subset",
                "in_balanced_subset_10",
                "first_turn_prompt",
                "user_prompt",
                "model_response",
                "turn",
            ],
            var_name="turn2",
            value_name="feedback",
        )
        df = (
            df[df["turn"] == df["turn2"]]
            .reset_index(drop=True)
            .drop(columns=["turn2"])
        )
        print(df)
        ds.to_pandas().to_pickle(f"data/cad_en_preprocessed.gz")
        df.to_pickle(f"data/cad_en_utterances_preprocessed.gz")
