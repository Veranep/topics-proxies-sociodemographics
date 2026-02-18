import numpy as np
import pandas as pd
import json
import argparse


def get_gender(score):
    if score < 3:
        return "Female"
    elif score > 3:
        return "Male"
    else:
        return np.nan


extracted_belief_mapping = {
    "Socioeconomic Status": [
        (
            [
                "Middle income",
                "Middle Class",
                "Middle class",
                "Middle to Upper-middle income",
                "Middle to Upper-middle class",
                "Middle to Upper Middle Income",
                "Lower-middle class",
            ],
            "Middle income",
        )
    ],
    "Marital Status": [
        (["Single"], "Never married"),
        (
            [
                "In a relationship",
                "Engaged",
                "In a romantic partnership",
                "In a non-monogamous relationship",
                "In a romantic relationship",
                "In a committed relationship or planning to get married",
            ],
            "In a relationship",
        ),
    ],
}

answer_detect_map = {
    "hehimhis": "male",
    "sheherhers": "female",
    "theythemtheirs": "non-binary",
    "non-binary": "non-binary",
    "they": "non-binary",
    "them": "non-binary",
    "their": "non-binary",
    "themselves": "non-binary",
    "female": "female",
    "male": "male",
    "she": "female",
    "her": "female",
    "herself": "female",
    "woman": "female",
    "womans": "female",
    "womens": "female",
    "girl": "female",
    "sheher": "female",
    "mans": "male",
    "mens": "male",
    "hehim": "male",
    "hehis": "male",
    "theythem": "non-binary",
    "theythey": "non-binary",
    "he": "male",
    "his": "male",
    "him": "male",
    "man": "male",
    "himself": "male",
    "boy": "male",
    "masculine": "male",
    "feminine": "female",
    "women": "female",
    "men": "male",
}

extracted_belief_options = {
    "Age": ["Young adult", "Middle-aged adult", "Older adult"],
    "Gender": ["Male", "Female", "Non-Binary"],
    "English proficiency": ["Native speaker", "Non-Native speaker"],
    "Ethnicity": [
        "Asian",
        "Black",
        "Hispanic",
        "White",
        "Māori or Pacific Islander",
    ],
    "Socioeconomic Status": [
        "Low income",
        "High income",
        "Middle income",
        "Middle Class",
        "Middle class",
        "Middle to Upper-middle income",
        "Middle to Upper-middle class",
        "Middle to Upper Middle Income",
        "Lower-middle class",
    ],
    "Educational Background": ["Low", "High"],
    "Marital Status": [
        "Never married",
        "Married",
        "Divorced",
        "Widowed",
        "Single",
        "In a relationship",
        "Engaged",
        "In a romantic partnership",
        "In a non-monogamous relationship",
        "In a romantic relationship",
        "In a committed relationship or planning to get married",
    ],
    "Religion": [
        "No Affiliation",
        "Christian",
        "Jewish",
        "Muslim",
        "Buddhist",
        "Hindu",
        "Sikh",
        "Pagan",
    ],
}

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
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    df_extracted_belief = pd.read_pickle(
        f"{args.data_folder}/{args.model.split('/')[1]}_{args.dataset}_extracted_belief.gz"
    )
    df_revealed_belief = pd.read_pickle(
        f"{args.data_folder}/{args.model.split('/')[1]}_{args.dataset}_answers.gz"
    )
    extracted_columns_unknown = [
        c for c in df_extracted_belief if "unknown_token" in c
    ]
    for c in extracted_columns_unknown:
        demographic = c.split("unknown_token_")[1]
        options = extracted_belief_options[demographic]
        df_extracted_belief[c] = df_extracted_belief[c].str.extract(
            f"({'|'.join(options +['Unknown'])})"
        )

    extracted_columns_JSON = [
        c for c in df_extracted_belief if "reason_JSON" in c
    ]
    for c in extracted_columns_JSON:
        demographic = c.split("reason_JSON_")[1]
        options = extracted_belief_options[demographic]
        df_extracted_belief[f"value_JSON_{demographic}"] = df_extracted_belief[
            c
        ].str.extract(f"({'|'.join(options)})")
        if demographic in extracted_belief_mapping:
            for mapping in extracted_belief_mapping[demographic]:
                df_extracted_belief[f"value_JSON_{demographic}"] = (
                    df_extracted_belief[f"value_JSON_{demographic}"].replace(
                        mapping[0], mapping[1]
                    )
                )

        df_extracted_belief[c] = (
            df_extracted_belief[c].str.split("Reasons", n=1).str[1]
        )

    df_revealed_belief["q_60"] = (
        df_revealed_belief["q_60"]
        .str.replace("/", " ")
        .str.translate(
            str.maketrans("", "", "!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
        )
        .str.lower()
        .str.split()
    )
    revealed_gender = []
    for i in range(len(df_revealed_belief)):
        options = []
        for token in answer_detect_map:
            if token in df_revealed_belief["q_60"].iloc[i]:
                options.append(answer_detect_map[token])
        revealed_gender.append("-".join(sorted(set(options))))
    df_revealed_belief["revealed_Gender"] = revealed_gender
    df_revealed_belief["revealed_Gender"] = df_revealed_belief[
        "revealed_Gender"
    ].replace("", np.nan)

    df = (
        df_extracted_belief.merge(
            df_revealed_belief,
            on=[
                c
                for c in df_revealed_belief
                if c in df_extracted_belief
                and c
                not in [
                    "style_score",
                    "conversation_history",
                    "performance_attributes",
                    "choice_attributes",
                    "location",
                    "lm_usecases",
                    "stated_prefs",
                    "order_lm_usecases",
                    "order_stated_prefs",
                ]
            ],
        )
        .drop(
            columns=[
                "q_0",
                "q_1",
                "q_2",
                "q_3",
                "q_4",
                "q_5",
                "q_6",
                "q_7",
                "q_8",
                "q_9",
                "q_10",
                "q_11",
                "q_12",
                "q_13",
                "q_14",
                "q_15",
                "q_16",
                "q_17",
                "q_18",
                "q_19",
                "q_20",
                "q_21",
                "q_22",
                "q_23",
                "q_24",
                "q_25",
                "q_26",
                "q_27",
                "q_28",
                "q_29",
                "q_30",
                "q_31",
                "q_32",
                "q_33",
                "q_34",
                "q_35",
                "q_36",
                "q_37",
                "q_38",
                "q_39",
                "q_40",
                "q_41",
                "q_42",
                "q_43",
                "q_44",
                "q_45",
                "q_46",
                "q_47",
                "q_48",
                "q_49",
                "q_50",
                "q_51",
                "q_52",
                "q_53",
                "q_54",
                "q_55",
                "q_56",
                "q_57",
                "q_58",
                "q_59",
                "q_60",
                "style_score_y",
                "conversation_history_y",
                "performance_attributes_y",
                "choice_attributes_y",
                "location_y",
                "lm_usecases_y",
                "stated_prefs_y",
                "order_lm_usecases_y",
                "order_stated_prefs_y",
            ],
            errors="ignore",
        )
        .rename(
            columns={
                "style_score_x": "style_score",
                "conversation_history_x": "conversation_history",
                "performance_attributes_x": "performance_attributes",
                "choice_attributes_x": "choice_attributes",
                "location_x": "location",
                "lm_usecases_x": "lm_usecases",
                "stated_prefs_x": "stated_prefs",
                "order_lm_usecases_x": "order_lm_usecases",
                "order_stated_prefs_x": "order_stated_prefs",
            }
        )
    )
    for demographic in extracted_belief_options:
        df[f"shared_extracted_{demographic}"] = (
            df[f"value_JSON_{demographic}"]
            * (
                df[f"value_JSON_{demographic}"]
                == df[f"unknown_token_{demographic}"]
            )
        ).replace("", np.nan)

    if args.dataset == "chen":
        df["human_Gender"] = [
            get_gender(score)
            for score in np.mean(df["style_score"].tolist(), axis=1)
        ]
        df = df.drop(columns=["style_score"])

    df.to_pickle(
        f"{args.data_folder}/{args.model.split('/')[1]}_{args.dataset}_beliefs_preprocessed.gz"
    )
