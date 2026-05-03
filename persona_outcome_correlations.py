import argparse
from copy import deepcopy
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

demographics = {
    "prism": [
        "age",
        "gender",
        "employment_status",
        "education",
        "marital_status",
        "english_proficiency",
        "religion",
        "ethnicity",
        "birth_region",
        "reside_region",
        "lm_familiarity",
    ],
    "cad_en": [
        "annotator_age",
        "annotator_gender",
        "annotator_education_level",
        "annotator_political",
        "annotator_ethnicity",
    ],
}

domains = ["legal", "salary", "medical", "benefits", "political"]

questions = pd.read_pickle(
    "/scratch/vneplen/sociodemographics-interpretability-mitigation/data/Llama-3.1-8B-Instruct_questions.gz"
)
questions_correct_answers = dict(zip(questions.q_id, questions.correct_answer))
domain_qid_map = {
    domain: questions.loc[questions["domain"] == domain, "q_id"].tolist()
    for domain in domains
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
    args = parser.parse_args()
    all_cols = deepcopy(demographics[args.dataset])
    if args.model.split("/")[1] == "Llama-3.1-8B-Instruct":
        df = pd.read_pickle(
            f"/scratch/vneplen/sociodemographics-interpretability-mitigation/behavior/{args.model.split('/')[1]}_{args.dataset}_answers.gz"
        )

        for c in [
            qid for d in domains for qid in domain_qid_map[d] if d != "salary"
        ]:
            df[c] = 1 * (df[c].str.lower() == questions_correct_answers[c])

        for c in domain_qid_map["salary"]:
            df[c] = (
                df[c]
                .str.replace(",", "")
                .str.extract(r"^[^\d]*(\d+)")
                .astype(float)
            )

        for domain in domains:
            df[domain] = df[[qid for qid in domain_qid_map[domain]]].mean(
                axis=1
            )
            if domain != "salary":
                df[domain] = df[domain] * 100

        df = df.drop(
            columns=[f"q_{i}" for i in range(50)]
            + ["q_59", "q_60"]
            + [f"q_{i}" for i in range(61, 211)]
        )

        for column in demographics[args.dataset]:
            df = df.loc[
                ~(df[column].isna())
                & (df[column] != "Prefer not to say")
                & (df[column] != "Other")
                & (df[column] != "Unknown")
            ]
            df[column] = pd.factorize(df[column])[0]

    for domain in tqdm(domains):
        if args.model.split("/")[1] != "Llama-3.1-8B-Instruct":
            df = pd.read_pickle(
                f"/scratch/vneplen/sociodemographics-interpretability-mitigation/behavior/{args.model.split('/')[1]}_{args.dataset}_{domain}_answers.gz"
            )

            for c in domain_qid_map[domain]:
                if domain != "salary":
                    df[c] = 1 * (
                        df[c].str.lower() == questions_correct_answers[c]
                    )
                else:
                    df[c] = (
                        df[c]
                        .str.replace(",", "")
                        .str.extract(r"^[^\d]*(\d+)")
                        .astype(float)
                    )

            df[domain] = df[[qid for qid in domain_qid_map[domain]]].mean(
                axis=1
            )
            if domain != "salary":
                df[domain] = df[domain] * 100

            df = df.drop(
                columns=[f"q_{i}" for i in range(50)]
                + ["q_59", "q_60"]
                + [f"q_{i}" for i in range(61, 211)]
            )

            for column in demographics[args.dataset]:
                df = df.loc[
                    ~(df[column].isna())
                    & (df[column] != "Prefer not to say")
                    & (df[column] != "Other")
                    & (df[column] != "Unknown")
                ]
                df[column] = pd.factorize(df[column])[0]

        dist_user = []
        dist_outcome = []
        for i in tqdm(range(len(df))):
            for j in range(len(df)):
                if i == j:
                    continue
                dist_user.append(
                    np.linalg.norm(
                        df[demographics[args.dataset]].iloc[i].to_numpy()
                        - df[demographics[args.dataset]].iloc[j].to_numpy()
                    )
                )
                dist_outcome.append(
                    abs(df[domain].iloc[i] - df[domain].iloc[j])
                )
        print(
            domain,
            spearmanr(dist_user, dist_outcome),
            pearsonr(dist_user, dist_outcome),
        )
