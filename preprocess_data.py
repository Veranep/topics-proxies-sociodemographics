import argparse
from datasets import load_dataset
import numpy as np
import pandas as pd
import pickle


def get_prism_turns(row):
    return [
        {
            "role": turn["role"].replace("model", "assistant"),
            "content": turn["content"],
        }
        for turn in row.conversation_history
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


def get_chen_turns(row):
    return [
        {
            "role": "user",
            "content": f"Could you paraphrase my writing: '{row.user_prompt}'?",
        }
    ]
    # return [
    #     [
    #         {
    #             "role": "user",
    #             "content": f"Could you paraphrase my writing: '{row.user_prompt}'?",
    #         }
    #     ],
    #     [
    #         {
    #             "role": "user",
    #             "content": f"Please fix any grammar or spelling mistakes in my writing: '{row.user_prompt}'.",
    #         }
    #     ],
    #     [
    #         {
    #             "role": "user",
    #             "content": f"What are a few good titles for my text: '{row.user_prompt}'?",
    #         }
    #     ],
    # ]


def get_chen_convos(df):
    convos = list(map(get_chen_turns, df.itertuples(index=False)))
    return convos
    # return [c for cs in convos for c in cs]


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
        df_utterances.to_pickle(
            f"{args.folder}/prism_utterances_preprocessed.gz"
        )

    elif args.dataset == "cad":
        ds = load_dataset(
            "facebook/community-alignment-dataset", split="filtered"
        )
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
                "user_prompt",
            ],
            var_name="turn",
            value_name="model_response",
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
                "user_prompt",
                "model_response",
            ],
            var_name="turn",
            value_name="feedback",
        )
        print(df)
        df.to_pickle(f"{args.folder}/cad_utterances_preprocessed.gz")

        for language in ["en", "fr", "it", "pt", "hi"]:
            filtered_ds = ds.filter(
                lambda example: example["assigned_lang"] == language
            )
            filtered_ds.to_pandas().to_pickle(
                f"{args.folder}/cad_{language}_preprocessed.gz"
            )
            lang_df = df[df["language"] == language].reset_index()
            lang_df.to_pickle(
                f"{args.folder}/cad_{language}_utterances_preprocessed.gz"
            )

    elif args.dataset == "chen":
        df = pd.read_csv(f"{args.folder}/responses.csv")
        df = df.rename(columns={"short_text": "user_prompt"})
        df = (
            df.groupby(["text_id", "user_prompt", "label"])["style_score"]
            .apply(list)
            .reset_index()
        )
        df.to_pickle(f"{args.folder}/chen_preprocessed.gz")
        print(get_chen_convos(df)[:5])
