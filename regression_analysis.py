#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load_ext jupyter_black


# In[2]:


from copy import deepcopy
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
import statsmodels.formula.api as smf

# from sklearn.preprocessing import minmax_scale

# import pyfixest as pf


# In[3]:


def balance_df(df, col):
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
        df.loc[df[col] == "White"] = 0
        df.loc[df[col] == "Black or African American"] = 1
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
    elif col == "ethnicity":
        df.loc[df[col] == "White", col] = 0
        df.loc[df[col] == "Black", col] = 1
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
    return selected_df


# In[4]:


questions = pd.read_pickle("data/Llama-3.1-8B-Instruct_questions.gz")
questions_correct_answers = dict(zip(questions.q_id, questions.correct_answer))
questions_baseline_answers = dict(
    zip(questions.q_id, questions.baseline_answer)
)


# In[5]:


options = {
    "q_50": ["1", "2", "3", "4"],
    "q_51": ["A", "B"],
    "q_52": ["1", "2", "3"],
    "q_53": ["A", "B", "C"],
    "q_54": ["10", "2", "3", "4", "5", "6", "7", "8", "9", "1"],
    "q_55": ["10", "2", "3", "4", "5", "6", "7", "8", "9", "1"],
    "q_56": ["10", "2", "3", "4", "5", "6", "7", "8", "9", "1"],
    "q_57": ["1", "2", "3", "4"],
    "q_58": [
        "1,1",
        "1,2",
        "1,3",
        "1,4",
        "2,1",
        "2,2",
        "2,3",
        "2,4",
        "3,1",
        "3,2",
        "3,3",
        "3,4",
        "4,1",
        "4,2",
        "4,3",
        "4,4",
    ],
    "q_59": [
        "Good manners",
        "Independence",
        "Hard work",
        "Feeling of responsibility",
        "Imagination",
        "Tolerance and respect for other people",
        "Thrift, saving money and things",
        "Determination",
        "Religious faith",
        "Not being selfish",
        "Obedience",
    ],
}

id_col = {
    "prism": "conversation_id",
    "chen": "text_id",
    "cad_en": "conversation_id",
    "cad_fr": "conversation_id",
    "cad_pt": "conversation_id",
    "cad_it": "conversation_id",
}

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
    "chen": [
        "Gender",
        "human_Gender",
    ],
    "cad_en": [
        "annotator_age",
        "annotator_gender",
        "annotator_education_level",
        "annotator_political",
        "annotator_ethnicity",
    ],
    "cad_fr": [
        "annotator_age",
        "annotator_gender",
        "annotator_education_level",
        "annotator_political",
        "annotator_ethnicity",
    ],
    "cad_pt": [
        "annotator_age",
        "annotator_gender",
        "annotator_education_level",
        "annotator_political",
        "annotator_ethnicity",
    ],
    "cad_it": [
        "annotator_age",
        "annotator_gender",
        "annotator_education_level",
        "annotator_political",
        "annotator_ethnicity",
    ],
}


# In[6]:


for dataset in [
    "cad_en",  # "chen",
    "prism",
]:
    all_cols = deepcopy(demographics[dataset])
    df = pd.read_pickle(
        f"llama_beliefs/Llama-3.1-8B-Instruct_{dataset}_answers.gz"
    )
    df = df.rename(columns={"label": "Gender"})
    for c in (
        # [f"q_{i}" for i in range(50)]
        # +
        [f"q_{i}" for i in range(61, 121)]
        + [f"q_{i}" for i in range(151, 211)]
    ):
        df[c] = 1 * (df[c].str.lower() == questions_correct_answers[c])

    for c in [f"q_{i}" for i in range(121, 151)]:
        df[c] = (
            df[c]
            .str.replace(",", "")
            .str.extract(r"^[^\d]*(\d+)")
            .astype(float)
        )

    for domain in ["benefits", "political", "salary", "legal", "medical"]:
        qrange = {
            "benefits": (61, 91),
            "political": (91, 121),
            "salary": (121, 151),
            "legal": (151, 181),
            "medical": (181, 211),
        }[domain]
        df = df.melt(
            id_vars=[
                c
                for c in df.columns
                if c not in [f"q_{qid}" for qid in range(*qrange)]
            ],
            var_name="Question",
            value_name=domain,
        )

    print(df)
    # df["accuracy"] = df[[f"q_{i}" for i in range(50)]].mean(axis=1)
    # df["benefits"] = df[[f"q_{i}" for i in range(61, 91)]].mean(axis=1)
    # df["political"] = df[[f"q_{i}" for i in range(91, 121)]].mean(axis=1)
    # df["legal"] = df[[f"q_{i}" for i in range(151, 181)]].mean(axis=1)
    # df["medical"] = df[[f"q_{i}" for i in range(181, 211)]].mean(axis=1)
    # df["salary"] = df[[f"q_{i}" for i in range(121, 151)]].mean(axis=1)

    for c in [
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
    ]:
        if c == "q_53":
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace({"A": 0, "B": 0.5, "C": 1})
            df[c] = df[c].astype(float)
        elif c == "q_51":
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace({"A": 0, "B": 1})
            df[c] = df[c].astype(float)
        elif c == "q_58":
            df[c] = df[c].str.replace(" ", "")
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace(
                {
                    "1,1": 0,
                    "1,3": 0,
                    "3,1": 0,
                    "3,3": 0,
                    "2,2": 1,
                    "1,2": 0.5,
                    "1,4": 0.5,
                    "2,3": 0.5,
                    "2,1": 0.5,
                    "3,2": 0.5,
                    "3,4": 0.5,
                    "4,1": 0.5,
                    "4,3": 0.5,
                    "2,4": 1,
                    "4,2": 1,
                    "4,4": 1,
                }
            )
            df[c] = df[c].astype(float)
        elif c == "q_59":
            for v in options[c]:
                df[f"{c}_{v.replace(' ','').replace(',','')}"] = (
                    df[c].str.contains(v).astype(float)
                )
        else:
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].astype(float)
    df = df.drop(
        columns=[f"q_{i}" for i in range(50)]
        + ["q_59", "q_60"]
        + [f"q_{i}" for i in range(61, 211)]
    )

    df_linguistic = pd.read_pickle(
        f"data/{dataset + '_utterances' if dataset != 'chen' else dataset}_linguistic.gz"
    ).drop(
        columns=["s_neutral_model_response", "s_neutral_user_prompt"],
        errors="ignore",
    )
    for c in ["politeness_user_prompt", "politeness_model_response"]:
        if c in df_linguistic:
            df_linguistic[c] = df_linguistic[c].replace(
                {
                    "impolite": 0,
                    "neutral": 0.5,
                    "polite": 1,
                    "somewhat polite": 0.75,
                }
            )
    df_linguistic = df_linguistic.rename(columns={"gpt_description": "topic"})
    if dataset != "chen":
        if dataset == "prism":
            all_cols += ["model_name"]
        all_cols += ["topic"]
        group_cols = [id_col[dataset]] + all_cols
        df_linguistic = (
            df_linguistic.groupby(group_cols)[
                [
                    c
                    for c in df_linguistic.columns
                    if "_model_response" in c or "_user_prompt" in c
                ]
            ]
            .mean()
            .reset_index()
        )
    df = df.merge(
        df_linguistic[
            [id_col[dataset]]
            + [
                c
                for c in df_linguistic.columns
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "topic"
                or c == "model_name"
            ]
        ],
        on=id_col[dataset],
    )
    all_cols += [
        c for c in df.columns if "_model_response" in c or "_user_prompt" in c
    ]

    df_beliefs = pd.read_pickle(
        f"llama_beliefs/Llama-3.1-8B-Instruct_{dataset}_beliefs_preprocessed.gz"
    )
    df_beliefs.columns = df_beliefs.columns.str.replace(" ", "")
    cols = [
        c
        for c in df_beliefs.columns
        if "shared_extracted_" in c
        or "value_JSON_" in c
        or "unknown_token_" in c
    ] + ["revealed_Gender"]
    if "human_Gender" in df_beliefs.columns:
        cols += ["human_Gender"]
    all_cols += cols
    cols.append(id_col[dataset])
    df = df.merge(df_beliefs[cols], on=id_col[dataset])

    # df[
    #     [
    #         c
    #         for c in all_cols
    #         if "_model_response" in c or "_user_prompt" in c
    #     ]
    # ] = minmax_scale(
    #     df[
    #         [
    #             c
    #             for c in all_cols
    #             if "_model_response" in c or "_user_prompt" in c
    #         ]
    #     ]
    # )

    print("df finished")

    cols = [
        # "accuracy",
        "benefits",
        "political",
        "legal",
        "medical",
        "salary",
        "q_50",
        "q_51",
        "q_52",
        "q_53",
        "q_54",
        "q_55",
        "q_56",
        "q_57",
        "q_58",
    ] + [f"q_59_{v.replace(' ','').replace(',','')}" for v in options["q_59"]]

    for demographic in demographics[dataset]:
        if not os.path.isfile(
            f"figures_regression/{dataset}_{demographic}_1.png"
        ):
            filtered_df_demo = balance_df(df, demographic)
            filtered_df_demo[demographic] = filtered_df_demo[
                demographic
            ].astype(float)
            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{demographic} ~ {' + '.join(demo_cols)}",
                data=filtered_df_demo,
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            fig = plt.figure(figsize=(6.5, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{demographic}_1.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())

            filtered_df_demo = filtered_df_demo.loc[
                filtered_df_demo.duplicated(subset=["topic"], keep=False)
            ]

            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "topic"
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{demographic} ~ {' + '.join(demo_cols)}",
                data=filtered_df_demo,
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            result_df["linguistic"] = ~result_df["index"].str.contains("topic")
            fig = plt.figure(figsize=(40, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
                hue="linguistic",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{demographic}_2.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())

    for col in cols:
        filtered_df = df.loc[~df[col].isna()]
        if not os.path.isfile(
            f"figures_regression/{dataset}_{col}_correct_1.png"
        ):
            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{col} ~ {' + '.join(demo_cols)}", data=filtered_df
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            fig = plt.figure(figsize=(6.5, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{col}_correct_1.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())

        if "topic" in filtered_df:
            filtered_df = filtered_df.loc[
                filtered_df.duplicated(subset=["topic"], keep=False)
            ]
        if not os.path.isfile(
            f"figures_regression/{dataset}_{col}_correct_2.png"
        ):
            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "topic"
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{col} ~ {' + '.join(demo_cols)}", data=filtered_df
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            result_df["linguistic"] = ~result_df["index"].str.contains("topic")
            fig = plt.figure(figsize=(40, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
                hue="linguistic",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{col}_correct_2.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())

        for demographic in [
            "age",
            "gender",
            "education",
            "ethnicity",
            "religion",
            "english",
            "marital",
            "political",
        ]:
            if not os.path.isfile(
                f"figures_regression/{dataset}_{col}_{demographic}_correct_3.png"
            ):
                demo_cols = [
                    c
                    for c in all_cols
                    if demographic in c.lower()
                    or "_model_response" in c
                    or "_user_prompt" in c
                    or c == "topic"
                    or c == "model_name"
                ]

                mod = smf.ols(
                    formula=f"{col} ~ {' + '.join(demo_cols)}",
                    data=filtered_df,
                )
                res = mod.fit()
                result_df = pd.read_html(
                    res.summary().tables[1].as_html(), header=0, index_col=0
                )[0].reset_index()
                result_df["type"] = result_df["index"].str.extract("(topic)")
                result_df["type"].loc[
                    result_df["index"].str.contains(demographic)
                ] = "demographic"
                result_df["type"].loc[result_df["type"].isna()] = "linguistic"
                fig = plt.figure(figsize=(45, 5))
                ax = sns.barplot(
                    result_df.loc[
                        (result_df["P>|t|"] < 0.05)
                        & (result_df["index"] != "Intercept")
                    ].sort_values(by="coef"),
                    x="index",
                    y="coef",
                    hue="type",
                )
                ax.tick_params(axis="x", labelrotation=90)
                fig.savefig(
                    f"figures_regression/{dataset}_{col}_{demographic}_correct_3.png",
                    bbox_inches="tight",
                )
                plt.show()
                print(res.summary())

                demo_cols = [
                    c
                    for c in all_cols
                    if demographic in c.lower()
                    or "_model_response" in c
                    or "_user_prompt" in c
                    or c == "model_name"
                ]


# In[ ]:


for dataset in [
    "cad_en",  # "chen",
    "prism",
]:
    all_cols = demographics[dataset]
    df = pd.read_pickle(
        f"llama_beliefs/Llama-3.1-8B-Instruct_{dataset}_answers.gz"
    )
    df = df.rename(columns={"label": "Gender"})
    for c in (
        # [f"q_{i}" for i in range(50)]
        # +
        [f"q_{i}" for i in range(61, 121)]
        + [f"q_{i}" for i in range(151, 211)]
    ):
        df[c] = 1 * (df[c].str.lower() == questions_baseline_answers[c])

    for c in [f"q_{i}" for i in range(121, 151)]:
        df[c] = (
            df[c]
            .str.replace(",", "")
            .str.extract(r"^[^\d]*(\d+)")
            .astype(float)
        )

    for domain in ["benefits", "political", "salary", "legal", "medical"]:
        qrange = {
            "benefits": (61, 91),
            "political": (91, 121),
            "salary": (121, 151),
            "legal": (151, 181),
            "medical": (181, 211),
        }[domain]
        df = df.melt(
            id_vars=[
                c
                for c in df.columns
                if c not in [f"q_{qid}" for qid in range(*qrange)]
            ],
            var_name="Question",
            value_name=domain,
        )

    print(df)
    # df["accuracy"] = df[[f"q_{i}" for i in range(50)]].mean(axis=1)
    # df["benefits"] = df[[f"q_{i}" for i in range(61, 91)]].mean(axis=1)
    # df["political"] = df[[f"q_{i}" for i in range(91, 121)]].mean(axis=1)
    # df["legal"] = df[[f"q_{i}" for i in range(151, 181)]].mean(axis=1)
    # df["medical"] = df[[f"q_{i}" for i in range(181, 211)]].mean(axis=1)
    # df["salary"] = df[[f"q_{i}" for i in range(121, 151)]].mean(axis=1)

    for c in [
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
    ]:
        if c == "q_53":
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace({"A": 0, "B": 0.5, "C": 1})
            df[c] = df[c].astype(float)
        elif c == "q_51":
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace({"A": 0, "B": 1})
            df[c] = df[c].astype(float)
        elif c == "q_58":
            df[c] = df[c].str.replace(" ", "")
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].replace(
                {
                    "1,1": 0,
                    "1,3": 0,
                    "3,1": 0,
                    "3,3": 0,
                    "2,2": 1,
                    "1,2": 0.5,
                    "1,4": 0.5,
                    "2,3": 0.5,
                    "2,1": 0.5,
                    "3,2": 0.5,
                    "3,4": 0.5,
                    "4,1": 0.5,
                    "4,3": 0.5,
                    "2,4": 1,
                    "4,2": 1,
                    "4,4": 1,
                }
            )
            df[c] = df[c].astype(float)
        elif c == "q_59":
            for v in options[c]:
                df[f"{c}_{v.replace(' ','').replace(',','')}"] = (
                    df[c].str.contains(v).astype(float)
                )
        else:
            df[c] = df[c].str.extract(f"({'|'.join(options[c])})")
            df[c] = df[c].astype(float)
    df = df.drop(
        columns=[f"q_{i}" for i in range(50)]
        + ["q_59", "q_60"]
        # + [f"q_{i}" for i in range(61, 121)]
        # + [f"q_{i}" for i in range(151, 211)]
    )

    df_linguistic = pd.read_pickle(
        f"data/{dataset + '_utterances' if dataset != 'chen' else dataset}_linguistic.gz"
    ).drop(
        columns=["s_neutral_model_response", "s_neutral_user_prompt"],
        errors="ignore",
    )
    for c in ["politeness_user_prompt", "politeness_model_response"]:
        if c in df_linguistic:
            df_linguistic[c] = df_linguistic[c].replace(
                {
                    "impolite": 0,
                    "neutral": 0.5,
                    "polite": 1,
                    "somewhat polite": 0.75,
                }
            )
    df_linguistic = df_linguistic.rename(columns={"gpt_description": "topic"})
    if dataset != "chen":
        if dataset == "prism":
            all_cols += ["model_name"]
        all_cols += ["topic"]
        group_cols = [id_col[dataset]] + all_cols
        df_linguistic = (
            df_linguistic.groupby(group_cols)[
                [
                    c
                    for c in df_linguistic.columns
                    if "_model_response" in c or "_user_prompt" in c
                ]
            ]
            .mean()
            .reset_index()
        )
    df = df.merge(
        df_linguistic[
            [id_col[dataset]]
            + [
                c
                for c in df_linguistic.columns
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "topic"
                or c == "model_name"
            ]
        ],
        on=id_col[dataset],
    )
    all_cols += [
        c for c in df.columns if "_model_response" in c or "_user_prompt" in c
    ]

    df_beliefs = pd.read_pickle(
        f"llama_beliefs/Llama-3.1-8B-Instruct_{dataset}_beliefs_preprocessed.gz"
    )
    df_beliefs.columns = df_beliefs.columns.str.replace(" ", "")
    cols = [
        c
        for c in df_beliefs.columns
        if "shared_extracted_" in c
        or "value_JSON_" in c
        or "unknown_token_" in c
    ] + ["revealed_Gender"]
    if "human_Gender" in df_beliefs.columns:
        cols += ["human_Gender"]
    all_cols += cols
    cols.append(id_col[dataset])
    df = df.merge(df_beliefs[cols], on=id_col[dataset])

    # df[
    #     [
    #         c
    #         for c in all_cols
    #         if "_model_response" in c or "_user_prompt" in c
    #     ]
    # ] = minmax_scale(
    #     df[
    #         [
    #             c
    #             for c in all_cols
    #             if "_model_response" in c or "_user_prompt" in c
    #         ]
    #     ]
    # )

    cols = [
        # "accuracy",
        "benefits",
        "political",
        "legal",
        "medical",
        "salary",
        "q_50",
        "q_51",
        "q_52",
        "q_53",
        "q_54",
        "q_55",
        "q_56",
        "q_57",
        "q_58",
    ] + [f"q_59_{v.replace(' ','').replace(',','')}" for v in options["q_59"]]

    for col in cols:
        filtered_df = df.loc[~df[col].isna()]
        if not os.path.isfile(
            f"figures_regression/{dataset}_{col}_baseline_1.png"
        ):
            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{col} ~ {' + '.join(demo_cols)}", data=filtered_df
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            fig = plt.figure(figsize=(6.5, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{col}_baseline_1.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())
        if "topic" in filtered_df:
            filtered_df = filtered_df.loc[
                filtered_df.duplicated(subset=["topic"], keep=False)
            ]
        if not os.path.isfile(
            f"figures_regression/{dataset}_{col}_baseline_2.png"
        ):
            demo_cols = [
                c
                for c in all_cols
                if "_model_response" in c
                or "_user_prompt" in c
                or c == "topic"
                or c == "model_name"
            ]
            mod = smf.ols(
                formula=f"{col} ~ {' + '.join(demo_cols)}", data=filtered_df
            )
            res = mod.fit()
            result_df = pd.read_html(
                res.summary().tables[1].as_html(), header=0, index_col=0
            )[0].reset_index()
            result_df["linguistic"] = ~result_df["index"].str.contains("topic")
            fig = plt.figure(figsize=(40, 5))
            ax = sns.barplot(
                result_df.loc[
                    (result_df["P>|t|"] < 0.05)
                    & (result_df["index"] != "Intercept")
                ].sort_values(by="coef"),
                x="index",
                y="coef",
                hue="linguistic",
            )
            ax.tick_params(axis="x", labelrotation=90)
            fig.savefig(
                f"figures_regression/{dataset}_{col}_baseline_2.png",
                bbox_inches="tight",
            )
            plt.show()
            print(res.summary())

        for demographic in [
            "age",
            "gender",
            "education",
            "ethnicity",
            "religion",
            "english",
            "marital",
            "political",
        ]:
            if not os.path.isfile(
                f"figures_regression/{dataset}_{col}_{demographic}_baseline_3.png"
            ):
                demo_cols = [
                    c
                    for c in all_cols
                    if demographic in c.lower()
                    or "_model_response" in c
                    or "_user_prompt" in c
                    or c == "topic"
                    or c == "model_name"
                ]

                mod = smf.ols(
                    formula=f"{col} ~ {' + '.join(demo_cols)}",
                    data=filtered_df,
                )
                res = mod.fit()
                result_df = pd.read_html(
                    res.summary().tables[1].as_html(), header=0, index_col=0
                )[0].reset_index()
                result_df["type"] = result_df["index"].str.extract("(topic)")
                result_df["type"].loc[
                    result_df["index"].str.contains(demographic)
                ] = "demographic"
                result_df["type"].loc[result_df["type"].isna()] = "linguistic"
                fig = plt.figure(figsize=(45, 5))
                ax = sns.barplot(
                    result_df.loc[
                        (result_df["P>|t|"] < 0.05)
                        & (result_df["index"] != "Intercept")
                    ].sort_values(by="coef"),
                    x="index",
                    y="coef",
                    hue="type",
                )
                ax.tick_params(axis="x", labelrotation=90)
                fig.savefig(
                    f"figures_regression/{dataset}_{col}_{demographic}_baseline_3.png",
                    bbox_inches="tight",
                )
                plt.show()
                print(res.summary())

                demo_cols = [
                    c
                    for c in all_cols
                    if demographic in c.lower()
                    or "_model_response" in c
                    or "_user_prompt" in c
                    or c == "model_name"
                ]
