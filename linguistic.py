import argparse
from datasets import load_dataset
import evaluate
import numpy as np
import spacy
import textstat
import torch
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import pickle

from compute_humt_sociot import calculate_td

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
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_pickle(
        f"{args.data_folder}/{args.dataset}{'_utterances' if args.dataset != 'chen' else ''}_preprocessed.gz"
    )
    if args.dataset == "prism":
        cluster_ids = pd.read_csv(
            f"{args.data_folder}/opening_prompt_text_df_original.csv"
        )[["id", "cluster_id"]]
        clusters = pd.read_csv(
            f"{args.data_folder}/opening_prompt_cluster_df_original.csv"
        )[["cluster_id", "gpt_description"]]
        cluster_ids = cluster_ids.merge(clusters).drop(columns=["cluster_id"])
        df = (
            df.merge(cluster_ids, left_on="conversation_id", right_on="id")
            .drop(columns=["id"])
            .rename({"gpt_description": "topic"})
        )
    elif "cad" in args.dataset:
        df["topic"] = df["first_turn_prompt"].loc[
            df["is_pregenerated_first_prompt"] == True
        ]
    perplexity = evaluate.load("perplexity", module_type="metric")
    if "cad" in args.dataset:
        language = args.dataset.split("_")[1]
    else:
        language = "en"
    # supported languages are 'en', 'it', 'pt', 'fr'
    if language == "en":
        nlp = spacy.load("en_core_web_sm")
        textstat.set_lang("en")
        emotion_classifier = pipeline(
            "text-classification",
            model="AnasAlokla/multilingual_go_emotions_V1.2",
            top_k=None,  # To return all scores for each label
            device=device,
            max_length=512,
            truncation=True,
        )
        politeness_classifier = pipeline(
            "text-classification",
            "Intel/polite-guard",
            device=device,
            max_length=512,
            truncation=True,
        )
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,  # To return all scores for each label
            device=device,
            max_length=512,
            truncation=True,
        )
        concreteness_df = pd.read_excel(
            f"{args.data_folder}/13428_2013_403_MOESM1_ESM.xlsx"
        )
        concreteness_dict = pd.Series(
            concreteness_df["Conc.M"].values, index=concreteness_df["Word"]
        ).to_dict()
    else:
        nlp = spacy.load(f"{language}_core_news_sm")
        if language in ["it", "fr"]:
            textstat.set_lang(language)
            if language == "fr":
                emotion_classifier = pipeline(
                    "text-classification",
                    model="AnasAlokla/multilingual_go_emotions",
                    top_k=None,  # To return all scores for each label
                    device=device,
                    max_length=512,
                    truncation=True,
                )
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    sentiments = ["negative", "neutral", "positive"]
    for column in ["user_prompt", "model_response"]:
        if column not in df:
            continue

        df[f"perplexity_{column}"] = perplexity.compute(
            model_id="ai-forever/mGPT",
            predictions=df[column].to_list(),
            device=device,
            max_length=512,
            batch_size=16 if language not in ["fr", "en"] else 8,
        )["perplexities"]

    del perplexity

    for column in ["user_prompt", "model_response"]:
        if column not in df:
            continue

        annotations = {
            "num_tokens": [],
            "num_sents": [],
            "num_unique_lemmas": [],
            "avg_sent_len": [],
            "type_to_token_ratio": [],
            "num_entities": [],
            "num_entities_per_sent": [],
        }
        if language in ["en", "it", "fr"]:
            annotations["avg_num_syllables"] = []
            annotations["flesch_reading_ease"] = []
            if language in ["en", "fr"]:
                for emotion in emotions:
                    annotations[emotion] = []
                if language == "en":
                    for sentiment in sentiments:
                        annotations[sentiment] = []
                    annotations["politeness"] = []
                    annotations["avg_concreteness"] = []
        for i in tqdm(range(len(df))):
            text = df.iloc[i][column]

            spacy_doc = nlp(text)
            num_sents = 0
            num_tokens = 0
            num_alpha_tokens = 0
            syllable_sum = 0
            concreteness_sum = 0
            unique_lemmas = set()
            for sent in spacy_doc.sents:
                num_sents += 1

                for token in sent:
                    if token.is_alpha:
                        num_alpha_tokens += 1
                        if language in ["en", "it", "fr"]:
                            syllable_sum += textstat.syllable_count(token.text)
                            if (
                                language == "en"
                                and token.text in concreteness_dict
                            ):
                                concreteness_sum += concreteness_dict[
                                    token.text
                                ]

                unique_lemmas.add(token.lemma_)

                num_tokens += 1

            avg_sent_len = (
                num_alpha_tokens / num_sents if num_sents > 0 else None
            )
            avg_num_syllables = (
                syllable_sum / num_alpha_tokens
                if num_alpha_tokens > 0
                else None
            )
            avg_concreteness = (
                concreteness_sum / num_alpha_tokens
                if num_alpha_tokens > 0
                else None
            )
            type_to_token_ratio = (
                len(unique_lemmas) / num_tokens if num_tokens > 0 else None
            )

            num_entities = len(spacy_doc.ents)
            num_entities_per_sent = (
                num_entities / num_sents if num_sents > 0 else None
            )

            if language in ["en", "it", "fr"]:
                reading_ease = textstat.flesch_reading_ease(text)
                annotations["flesch_reading_ease"].append(reading_ease)
                annotations["avg_num_syllables"].append(avg_num_syllables)
                if language in ["en", "fr"]:
                    results = emotion_classifier(text)
                    for entry in results[0]:
                        annotations[entry["label"]].append(entry["score"])
                    if language == "en":
                        politeness = politeness_classifier(text)[0]["label"]
                        annotations["politeness"].append(politeness)
                        results = sentiment_classifier(text)
                        for entry in results[0]:
                            annotations[entry["label"]].append(entry["score"])
                        annotations["avg_concreteness"].append(
                            avg_concreteness
                        )

            annotations["num_tokens"].append(num_tokens)
            annotations["num_sents"].append(num_sents)
            annotations["num_unique_lemmas"].append(len(unique_lemmas))
            annotations["avg_sent_len"].append(avg_sent_len)
            annotations["type_to_token_ratio"].append(type_to_token_ratio)
            annotations["num_entities"].append(num_entities)
            annotations["num_entities_per_sent"].append(num_entities_per_sent)

        for annotation in annotations:
            df[f"{annotation}_{column}"] = annotations[annotation]

        # if language == "en":
        #     for metric in tqdm(
        #         [
        #             "humt",
        #             "sociot_status",
        #             "sociot_social_distance",
        #             "sociot_gender",
        #             "sociot_warmth",
        #         ]
        #     ):
        #         df = calculate_td(df, column, metric)

    df.to_pickle(
        f"{args.data_folder}/{args.dataset}{'_utterances' if args.dataset != 'chen' else ''}_linguistic.gz"
    )
