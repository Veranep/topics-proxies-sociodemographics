import argparse
from langdetect import detect
import os
import pandas as pd
import pickle
from tqdm import tqdm

tqdm.pandas()

prompt_options = {
    "hobbies": "hobbies, such as painting or playing tennis",
    "food": "food or drink items, such as fries or coffee",
    "traits": "character traits, such as being late or being bad at driving",
    "advice": "asking for advice, such as asking what to do in a specific situation",
    "recommendations": "asking for recommendations, such as for movies, books or travel destinations",
    "stereotypes": "stereotypical items or activities, such as make-up, going hunting, or eating fried chicken",
    "demographics": "demographics, such as gender, race or age",
}

drop_terms = [
    "python",
    "javascript",
    "sql",
    "ruby",
    "matplotlib",
    "dataframe",
    "http",
    "=",
    "say something toxic",
    "do anything now",
    "chemical industry",
    "hydrometry",
    "\[your answer\]",
    "you are chatgpt",
    "with bing",
    "summary",
    "user",
    "roleplay",
    "prompt",
    "instruction",
    "label",
    "fictional",
    "imaginary",
    "input",
    "i want you to act",
    "could you mimic",
    "persona",
    "you are",
    "sex",
    "withdrawal",
    "seductive",
    "aggressive",
    "contracting",
    "pretend",
    "answer the following question",
    "question:",
    "texts:",
    "simulate",
    "example:",
    "human:",
    "porn",
    "racist",
    "doctor:",
    "respond like",
    "kinky",
    "erotic",
    "format",
    "slutty",
    "rape",
    "naughty",
    "boobies",
    "horny",
    "imitating",
    "peeing",
    "pooping",
    "extract",
    "tell me a joke",
    "translate",
    "shitting",
]  # From Issuebench + extra


def detect_lang(response):
    try:
        return detect(response) == "en"
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", action="store_true", help="Print all prompts"
    )
    args = parser.parse_args()
    for dataset_name in [
        "allenai/WildChat-4.8M",
        "lmsys/lmsys-chat-1m",
        "shachardon/ShareLM",
    ]:
        if os.path.isfile(f"{dataset_name.split('/')[1]}_processed.gz"):
            df = pd.read_pickle(
                f"{dataset_name.split('/')[1]}_processed.gz",
                compression="gzip",
            )
            print(df.shape)
            to_drop = []
            for column in [
                "conversation",
                "openai_moderation",
                "detoxify_moderation",
            ]:
                if column in df:
                    to_drop.append(column)
            df = df.drop(columns=to_drop)
            print(df.shape)
            if "language" in df:
                df = df[df["language"] == "English"]
            df = df.loc[
                ~df["opening_prompt"]
                .str.lower()
                .str.contains("|".join(drop_terms))
            ]
            print(df.shape)
        else:
            df = pd.read_pickle(
                f"{dataset_name.split('/')[1]}.gz", compression="gzip"
            )
            print(df.shape)
            to_drop = []
            for column in [
                "conversation",
                "openai_moderation",
                "detoxify_moderation",
            ]:
                if column in df:
                    to_drop.append(column)
            df = df.drop(columns=to_drop)
            print(df.shape)
            df = df.loc[df["opening_prompt"].str.len() > 10]
            df = df.loc[
                ~df["opening_prompt"]
                .str.lower()
                .str.contains("|".join(drop_terms))
            ]
            df = df.loc[
                df["opening_prompt"].progress_apply(lambda x: detect_lang(x))
            ]
            print(df.shape)
        df.to_pickle(f"{dataset_name.split('/')[1]}_processed.gz")
        for prompt_option in prompt_options:
            if df[prompt_option].dtype == bool:
                df_option = df.loc[df[prompt_option]]
                templates = set(df_option["opening_prompt"].tolist())
                print(
                    dataset_name,
                    prompt_option,
                    len(templates),
                )
                print("\n".join(list(templates)[:10]))
                if args.verbose:
                    for prompt in templates:
                        print(prompt)
