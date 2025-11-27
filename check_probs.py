import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind


def get_binary_subset(df, col):
    df_copy = df
    if col == "age":
        df_copy.loc[df[col] == "18-24 years old", col] = 0
        df_copy.loc[df[col] == "55-64 years old", col] = 1
        df_copy.loc[df_copy[col] == "65+ years old", col] = 1
    elif col == "gender":
        df_copy.loc[df_copy[col] == "Male", col] = 0
        df_copy.loc[df_copy[col] == "Female", col] = 1
    elif col == "religion":
        df_copy.loc[df_copy[col] == "No Affiliation", col] = 0
        df_copy.loc[df_copy[col] == "Christian", col] = 1
        df_copy.loc[df_copy[col] == "Jewish", col] = 1
        df_copy.loc[df_copy[col] == "Muslim", col] = 1
    elif col == "ethnicity":
        df_copy.loc[df_copy[col] == "White", col] = 0
        df_copy.loc[df_copy[col] == "Hispanic", col] = 1
        df_copy.loc[df_copy[col] == "Black", col] = 1
        df_copy.loc[df_copy[col] == "Asian", col] = 1
        df_copy.loc[df_copy[col] == "Mixed", col] = 1
    elif col == "employment_status":
        df_copy.loc[df_copy[col] == "Unemployed, seeking work", col] = 0
        df_copy.loc[df_copy[col] == "Unemployed, not seeking work", col] = 0
        df_copy.loc[df_copy[col] == "Homemaker / Stay-at-home parent", col] = 0
        df_copy.loc[df_copy[col] == "Working full-time", col] = 1
    elif col == "education":
        df_copy.loc[df_copy[col] == "Some Primary", col] = 0
        df_copy.loc[df_copy[col] == "Completed Primary School", col] = 0
        df_copy.loc[df_copy[col] == "Some Secondary", col] = 0
        df_copy.loc[df_copy[col] == "Completed Secondary School", col] = 0
        df_copy.loc[df_copy[col] == "Graduate / Professional degree", col] = 1
    elif col == "birth_region":
        df_copy.loc[df_copy[col] == "Europe", col] = 0
        df_copy.loc[df_copy[col] == "Americas", col] = 1
    elif col == "reside_region":
        df_copy.loc[df_copy[col] == "Europe", col] = 0
        df_copy.loc[df_copy[col] == "Americas", col] = 1
    elif col == "marital_status":
        df_copy.loc[df_copy[col] == "Never been married", col] = 0
        df_copy.loc[df_copy[col] == "Married", col] = 1
    elif col == "english_proficiency":
        df_copy.loc[df_copy[col] == "Native speaker", col] = 0
        df_copy.loc[df_copy[col] == "Advanced", col] = 1
        df_copy.loc[df_copy[col] == "Intermediate", col] = 1
        df_copy.loc[df_copy[col] == "Basic", col] = 1
    return df_copy[df_copy[col].isin([0, 1])].reset_index(drop=True)


if __name__ == "__main__":
    demographics = [
        "age",
        "gender",
        "religion",
        "ethnicity",
        "employment_status",
        "education",
        "birth_region",
        "reside_region",
        "marital_status",
        "english_proficiency",
    ]
    olmo_climatefever = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/OLMo-2-1124-7B-Instruct_answers_climate_fever.gz",
        compression="gzip",
    )[["question", "conversation_id", "gold_answer", "answer"] + demographics]
    olmo_climatefever1 = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/OLMo-2-1124-7B-Instruct_answers_climate_fever_1.gz",
        compression="gzip",
    )[
        ["question", "conversation_id", "gold_answer", "answer", "probs"]
        + demographics
    ]
    print("Finished 1")
    olmo_climatefever2 = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/OLMo-2-1124-7B-Instruct_answers_climate_fever_2.gz",
        compression="gzip",
    )[
        ["question", "conversation_id", "gold_answer", "answer", "probs"]
        + demographics
    ]
    print("Finished 2")
    olmo_climatefever3 = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/OLMo-2-1124-7B-Instruct_answers_climate_fever_3.gz",
        compression="gzip",
    )[
        ["question", "conversation_id", "gold_answer", "answer", "probs"]
        + demographics
    ]
    print("Finished 3")
    olmo_climatefever4 = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/OLMo-2-1124-7B-Instruct_answers_climate_fever_4.gz",
        compression="gzip",
    )[
        ["question", "conversation_id", "gold_answer", "answer", "probs"]
        + demographics
    ]
    print("Finished 4")
    merged_climate_fever = pd.concat(
        [
            olmo_climatefever1,
            olmo_climatefever2,
            olmo_climatefever3,
            olmo_climatefever4,
        ],
        ignore_index=True,
    )
    print("Finished merge")

    dfs = {"OLMo Climate Fever": olmo_climatefever}
    baseline_acc = {}

    for df in dfs:
        baseline_acc[df] = (
            1
            * (
                dfs[df][dfs[df]["age"] == ""]["answer"].str.strip()
                == dfs[df][dfs[df]["age"] == ""]["gold_answer"]
            )
        ).sum() / len(dfs[df][dfs[df]["age"] == ""]["answer"])
        no_context_answers = {
            q: a
            for q, a in dfs[df][dfs[df]["age"] == ""][
                ["question", "answer"]
            ].values.tolist()
        }
        dfs[df]["answer_nocontext"] = 1 * (
            dfs[df]["question"].map(no_context_answers)
            == dfs[df]["answer"].str.strip()
        )
        dfs[df]["answer_correct"] = 1 * (
            dfs[df]["answer"].str.strip() == dfs[df]["gold_answer"]
        )
        dfs[df]["answer_no"] = 1 * (dfs[df]["answer"].str.strip() == "no")
        dfs[df]["gold_answer_no"] = 1 * (dfs[df]["gold_answer"] == "no")
        dfs[df] = dfs[df][dfs[df]["age"] != ""]
        dfs[df] = dfs[df].melt(
            id_vars=[
                "conversation_id",
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
                "question",
                "gold_answer",
                "answer",
            ],
            var_name="type",
            value_name="score",
        )
        model_df = dfs[df]
        unique_questions = model_df.sort_values("question")[
            "question"
        ].unique()
        n_questions = len(unique_questions)
        diff_questions = {}
        for demographic in tqdm(demographics):
            df = get_binary_subset(model_df, demographic)
            values_no = []
            values_acc = []
            values_nc = []
            for group in df[demographic].unique():
                group_values_no = (
                    df[df["type"] == "answer_no"][df[demographic] == group]
                    .sort_values(["conversation_id", "question"])["score"]
                    .tolist()
                )
                group_values_acc = (
                    df[df["type"] == "answer_correct"][
                        df[demographic] == group
                    ]
                    .sort_values(["conversation_id", "question"])["score"]
                    .tolist()
                )
                group_values_nc = (
                    df[df["type"] == "answer_nocontext"][
                        df[demographic] == group
                    ]
                    .sort_values(["conversation_id", "question"])["score"]
                    .tolist()
                )
                values_no.append(
                    [
                        group_values_no[i : i + n_questions]
                        for i in range(0, len(group_values_no), n_questions)
                    ]
                )
                values_acc.append(
                    [
                        group_values_acc[i : i + n_questions]
                        for i in range(0, len(group_values_acc), n_questions)
                    ]
                )
                values_nc.append(
                    [
                        group_values_nc[i : i + n_questions]
                        for i in range(0, len(group_values_nc), n_questions)
                    ]
                )
            pvalues = ttest_ind(*values_no).pvalue
            questions_diff = unique_questions[np.where(pvalues < 0.05)[0]]
            diff_questions[demographic] = questions_diff
            print("got questions")

            merged_climate = get_binary_subset(
                merged_climate_fever, demographic
            )
            all_data = {}
            demo = {}
            anti_demo = {}
            for _, row in merged_climate.iterrows():
                for n in row["answer"]:
                    if n not in demo:
                        demo[n] = [[], [], []]
                        anti_demo[n] = [[], [], []]
                        all_data[n] = [[], [], []]

                    all_data[n][0].append(row["answer"][n][demographic][0][0])
                    all_data[n][1].append(row["answer"][n][demographic][0][1])
                    all_data[n][2].append(
                        1
                        * (
                            np.argmax(row["answer"][n][demographic][0])
                            == row[demographic]
                        )
                    )
                    if row["question"] in questions_diff:
                        demo[n][0].append(row["answer"][n][demographic][0][0])
                        demo[n][1].append(row["answer"][n][demographic][0][1])
                        demo[n][2].append(
                            1
                            * (
                                np.argmax(row["answer"][n][demographic][0])
                                == row[demographic]
                            )
                        )
                    else:
                        anti_demo[n][0].append(
                            row["answer"][n][demographic][0][0]
                        )
                        anti_demo[n][1].append(
                            row["answer"][n][demographic][0][1]
                        )
                        anti_demo[n][2].append(
                            1
                            * (
                                np.argmax(row["answer"][n][demographic][0])
                                == row[demographic]
                            )
                        )

            overall_acc = []

            for n in demo:
                overall_acc.append(np.mean(all_data[n][2]))
                print(
                    demographic,
                    n,
                    (np.mean(demo[n][0]), np.mean(demo[n][1])),
                    (np.mean(anti_demo[n][0]), np.mean(anti_demo[n][1])),
                    (np.mean(all_data[n][0]), np.mean(all_data[n][1])),
                    ttest_ind(demo[n][0], anti_demo[n][0], axis=None).pvalue,
                    ttest_ind(demo[n][0], all_data[n][0], axis=None).pvalue,
                    np.mean(demo[n][2]),
                    np.mean(anti_demo[n][2]),
                    np.mean(all_data[n][2]),
                    ttest_ind(demo[n][2], anti_demo[n][2], axis=None).pvalue,
                    ttest_ind(demo[n][2], all_data[n][2], axis=None).pvalue,
                )
            print(np.sort(overall_acc), np.argsort(overall_acc))
