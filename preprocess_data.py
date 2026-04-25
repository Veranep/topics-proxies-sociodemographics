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


def get_prism_turns(row):
    turns = [
        {
            "role": turn["role"].replace("model", "assistant"),
            "content": turn["content"],
        }
        for turn in row.conversation_history
        if turn["role"] == "user" or turn["if_chosen"] == True
    ]
    return turns


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


def get_personamem_convos(df):
    return df["conversation_history"].tolist()


def get_cad_turns(row):
    turns = []
    for turn in ["first", "second", "third", "fourth"]:
        pref_response = f"{turn}_turn_preferred_response"
        if not getattr(row, pref_response):
            return turns
        else:
            try:
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
            except:
                print(getattr(row, pref_response))
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
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Folder to save dataset in",
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
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

        if not os.path.isfile(f"{args.folder}/prism_preprocessed.gz"):
            df.to_pickle(f"{args.folder}/prism_preprocessed.gz")
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
        if not os.path.isfile(
            f"{args.folder}/prism_utterances_preprocessed.gz"
        ):
            df_utterances.to_pickle(
                f"{args.folder}/prism_utterances_preprocessed.gz"
            )

    elif args.dataset == "cad":
        ds = load_dataset(
            "facebook/community-alignment-dataset", split="filtered"
        )
        if not os.path.isfile(f"{args.folder}/cad_preprocessed.gz"):
            ds.to_pandas().to_pickle(f"{args.folder}/cad_preprocessed.gz")
        print(get_cad_convos(ds.to_pandas())[:5])

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
        if not os.path.isfile(f"{args.folder}/cad_utterances_preprocessed.gz"):
            df.to_pickle(f"{args.folder}/cad_utterances_preprocessed.gz")

        for language in ["en", "fr", "it", "pt", "hi"]:
            filtered_ds = ds.filter(
                lambda example: example["assigned_lang"] == language
            )
            if not os.path.isfile(
                f"{args.folder}/cad_{language}_preprocessed.gz"
            ):
                filtered_ds.to_pandas().to_pickle(
                    f"{args.folder}/cad_{language}_preprocessed.gz"
                )
            lang_df = df[df["assigned_lang"] == language].reset_index()
            if not os.path.isfile(
                f"{args.folder}/cad_{language}_utterances_preprocessed.gz"
            ):
                lang_df.to_pickle(
                    f"{args.folder}/cad_{language}_utterances_preprocessed.gz"
                )

    elif args.dataset == "personamem":
        demographic_cols = [
            "age",
            "gender",
            "race",
        ]
        if len(os.listdir("data/personamem")) == 0:
            snapshot_download(
                repo_id="bowen-upenn/PersonaMem-v2",
                repo_type="dataset",
                allow_patterns=["data/raw_data/**"],
                local_dir="data",
            )
        else:
            data = {
                "age": [],
                "gender": [],
                "race": [],
                "conversation_id": [],
                "conversation_history": [],
                "topic": [],
            }
            for file_name in tqdm(os.listdir("data/personamem")):
                df = pd.read_json(f"data/personamem/{file_name}").T
                df = df.reset_index(names="conversation_id")

                all_conversations = []
                for k, v in df.iloc[0]["conversations"].items():
                    if k != "translation":
                        all_conversations.extend(v)

                for i, conversation in enumerate(all_conversations):
                    if "topic_query" not in conversation:
                        continue
                    for d in demographic_cols:
                        if d in df:
                            if d != "age":
                                data[d].append(
                                    df[d]
                                    .astype("string")
                                    .str.lower()
                                    .str.strip()
                                    .iloc[0]
                                )
                            else:
                                data[d].append(int(df[d].iloc[0]))

                        else:
                            if d == "race":
                                alternative_found = False
                                for alternative in [
                                    "racial_ethnic_identity",
                                    "race_ethnicity",
                                ]:
                                    if alternative in df:
                                        data[d].append(
                                            df[alternative]
                                            .astype("string")
                                            .str.lower()
                                            .str.strip()
                                        )
                                        alternative_found = True
                            if d != "race" or not alternative_found:
                                data[d].append(None)
                    data["conversation_id"].append(
                        df["conversation_id"].astype(str) + f"_{i}"
                    )
                    data["conversation_history"].append(
                        conversation["conversations"]
                    )
                    data["topic"].append(
                        conversation["topic_query"].lower().strip()
                    )
            counter_topics = Counter(data["topic"])
            interesting_topics = [
                c
                for c in counter_topics
                if counter_topics[c] >= 20 and c != "empty"
            ]
            print(interesting_topics)
            print(
                np.sum(
                    [min(counter_topics[i], 75) for i in interesting_topics]
                )
            )

            final_data = {
                "age": [],
                "gender": [],
                "race": [],
                "conversation_id": [],
                "conversation_history": [],
                "topic": [],
            }
            topic_counter = {c: 0 for c in interesting_topics}
            ids = list(range(len(data["conversation_id"])))
            np.random.shuffle(ids)
            for i in tqdm(ids):
                if (
                    data["topic"][i] in topic_counter
                    and topic_counter[data["topic"][i]] < 75
                ):
                    for d in data:
                        final_data[d].append(data[d][i])
                    topic_counter[data["topic"][i]] += 1
            df = pd.DataFrame(final_data)
            for d in demographic_cols:
                if d != "age":
                    df[d] = df[d].astype("string")

            df["age"] = [
                (
                    None
                    if pd.isna(a)
                    else (
                        "18-24"
                        if a >= 18 and a < 25
                        else (
                            "25-34"
                            if a >= 25 and a < 35
                            else (
                                "35-44"
                                if a >= 35 and a < 45
                                else (
                                    "45-54"
                                    if a >= 45 and a < 55
                                    else (
                                        "55-64"
                                        if a >= 55 and a < 65
                                        else "65+" if a >= 65 else None
                                    )
                                )
                            )
                        )
                    )
                )
                for a in df["age"]
            ]
            df["gender"] = [
                (
                    "Female"
                    if re.search(r"woman|female|girl", str(g), re.IGNORECASE)
                    else (
                        "Male"
                        if re.search(r"man|male|boy", str(g), re.IGNORECASE)
                        else "Non-binary / third gender"
                    )
                )
                for g in df["gender"]
            ]
            df["race"] = [
                (
                    None
                    if re.search(r"mixed|multiethnic", str(e), re.IGNORECASE)
                    else (
                        "White"
                        if re.search(
                            r"white|ashkenazi|eastern european",
                            str(e),
                            re.IGNORECASE,
                        )
                        else (
                            "Hispanic"
                            if re.search(
                                r"latin|hispanic|mexican|puerto rican",
                                str(e),
                                re.IGNORECASE,
                            )
                            else (
                                "Asian"
                                if re.search(
                                    r"asian|chinese|japanese|thai|indonesian|korean|japonesa|rohingya",
                                    str(e),
                                    re.IGNORECASE,
                                )
                                else (
                                    "Black"
                                    if re.search(
                                        r"black|afro|african",
                                        str(e),
                                        re.IGNORECASE,
                                    )
                                    else None
                                )
                            )
                        )
                    )
                )
                for e in df["race"]
            ]
            df = df.rename(columns={"race": "ethnicity"})
            print(get_personamem_convos(df)[:5])
            if not os.path.isfile(f"{args.folder}/personamem_preprocessed.gz"):
                df.to_pickle(f"{args.folder}/personamem_preprocessed.gz")

            final_data = {
                "age": [],
                "gender": [],
                "ethnicity": [],
                "conversation_id": [],
                "user_prompt": [],
                "model_response": [],
                "topic": [],
            }

            for row in df.itertuples(index=False):
                for t, turn in enumerate(row.conversation_history):
                    if turn["role"] == "user":
                        final_data["age"].append(row.age)
                        final_data["gender"].append(row.gender)
                        final_data["ethnicity"].append(row.ethnicity)
                        final_data["conversation_id"].append(
                            row.conversation_id.iloc[0]
                        )
                        final_data["user_prompt"].append(turn["content"])
                        if (t + 1) < len(row.conversation_history):
                            final_data["model_response"].append(
                                row.conversation_history[t + 1]["content"]
                            )
                        else:
                            final_data["model_response"].append(None)
                        final_data["topic"].append(row.topic)
            df = pd.DataFrame(final_data)
            if not os.path.isfile(
                f"{args.folder}/personamem_utterances_preprocessed.gz"
            ):
                df.to_pickle(
                    f"{args.folder}/personamem_utterances_preprocessed.gz"
                )
