import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def plot_feature_effects(demographic, n):
    # learned coefficients weighted by frequency of appearance
    average_feature_effects = (
        clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    )
    return (
        feature_names[np.argsort(average_feature_effects)[:n]].tolist(),
        feature_names[np.argsort(average_feature_effects)[-n:][::-1]].tolist(),
    )


if __name__ == "__main__":
    accuracies = {}
    for model in [
        "gemma-2-9b-it",
        "Llama-3.1-8B-Instruct",
        "Olmo-3-7B-Instruct",
        "OLMo-2-1124-7B-Instruct",
    ]:
        accuracies[model] = {"all": {}}
        big_df = pd.DataFrame()
        for dataset in ["health_misinfo", "climate_fever", "pubhealth"]:
            if not os.path.isfile(f"data/{model}_answers_{dataset}_full.gz"):
                continue
            accuracies[model][dataset] = {}
            results = pd.read_pickle(f"data/{model}_answers_{dataset}_full.gz")
            big_df = pd.concat([big_df, results])
            for demographic in demographics:
                all_feature_names = [set(), set()]
                accs = []
                selected_df = get_binary_subset(
                    results,
                    demographic,
                )
                for _ in range(5):
                    vectorizer = TfidfVectorizer(
                        sublinear_tf=True,
                        max_df=0.5,
                        min_df=5,
                        stop_words="english",
                    )
                    df_train, df_test = train_test_split(
                        selected_df, shuffle=True
                    )
                    X_train = vectorizer.fit_transform(df_train["answer"])
                    X_test = vectorizer.transform(df_test["answer"])
                    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
                    clf.fit(X_train, df_train[demographic].astype("int"))
                    pred = clf.predict(X_test)
                    print(
                        model,
                        demographic,
                        dataset,
                        df_test[demographic].astype("int").value_counts(),
                    )
                    acc = accuracy_score(
                        df_test[demographic].astype("int"), pred
                    )
                    accs.append(acc)
                    target_names = ["0", "1"]
                    feature_names = vectorizer.get_feature_names_out()
                    features = plot_feature_effects(demographic, 50)
                    all_feature_names[0].update(features[0])
                    all_feature_names[1].update(features[1])
                accuracies[model][dataset][demographic] = accs
                print(model, dataset, demographic, np.mean(accs))
                print(all_feature_names[0])
                print(all_feature_names[1])
        for demographic in demographics:
            all_feature_names = [set(), set()]
            accs = []
            selected_df = get_binary_subset(
                big_df,
                demographic,
            )
            for _ in range(5):
                vectorizer = TfidfVectorizer(
                    sublinear_tf=True,
                    max_df=0.5,
                    min_df=5,
                    stop_words="english",
                )
                df_train, df_test = train_test_split(selected_df, shuffle=True)
                X_train = vectorizer.fit_transform(df_train["answer"])
                X_test = vectorizer.transform(df_test["answer"])
                clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
                clf.fit(X_train, df_train[demographic].astype("int"))
                pred = clf.predict(X_test)
                acc = accuracy_score(df_test[demographic].astype("int"), pred)
                accs.append(acc)
                target_names = ["0", "1"]
                feature_names = vectorizer.get_feature_names_out()
                features = plot_feature_effects(demographic, 50)
                all_feature_names[0].update(features[0])
                all_feature_names[1].update(features[1])
            accuracies[model]["all"][demographic] = accs
            print(model, dataset, demographic, np.mean(accs))
            print(all_feature_names[0])
            print(all_feature_names[1])
    with open("data/class_acc_generations.pkl", "wb") as outfile:
        pickle.dump(accuracies, outfile)
