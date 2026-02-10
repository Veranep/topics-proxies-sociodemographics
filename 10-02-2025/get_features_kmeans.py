import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def plot_feature_effects(n):
    # learned coefficients weighted by frequency of appearance
    average_feature_effects = (
        clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    )
    return (
        feature_names[np.argsort(average_feature_effects)[:n]].tolist(),
        feature_names[np.argsort(average_feature_effects)[-n:][::-1]].tolist(),
    )


if __name__ == "__main__":

    health_misinfo = pd.read_pickle(
        "data/prism_questions_health_misinfo.gz",
        compression="gzip",
    )
    health_misinfo = (
        health_misinfo.groupby(["conversation_id"]).first().reset_index()
    )

    health_misinfo = health_misinfo[health_misinfo["conversation_id"] != ""]

    with open(
        "kmeans/Llama-3.1-8B-Instruct_31_prompt_question_kmeans2_results.pkl",
        "rb",
    ) as infile:
        k_means = pickle.load(infile)[2]

    cid_to_label = {"": pd.NA}

    for i in range(len(k_means)):
        cid_to_label[f"c{i}"] = k_means[i]

    health_misinfo["gender_labels"] = health_misinfo["conversation_id"].map(
        cid_to_label
    )
    all_feature_names = [set(), set()]
    accs = []
    f1 = []
    rocauc = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(
        health_misinfo, health_misinfo["gender_labels"].astype("int").values
    ):
        df_train = health_misinfo.iloc[train_index]
        df_test = health_misinfo.iloc[test_index]
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.5,
            min_df=5,
            stop_words="english",
        )
        X_train = vectorizer.fit_transform(
            df_train["conversation_history"].map(
                lambda x: " ".join([t["content"] for t in x])
            )
        )
        X_test = vectorizer.transform(
            df_test["conversation_history"].map(
                lambda x: " ".join([t["content"] for t in x])
            )
        )
        clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
        clf.fit(X_train, df_train["gender_labels"].astype("int"))
        pred = clf.predict(X_test)
        accs.append(
            accuracy_score(df_test["gender_labels"].astype("int"), pred)
        )
        f1.append(
            f1_score(
                df_test["gender_labels"].astype("int"),
                pred,
                average="weighted",
            )
        )
        rocauc.append(
            roc_auc_score(
                df_test["gender_labels"].astype("int"),
                clf.decision_function(X_test),
            )
        )
        target_names = ["0", "1"]
        feature_names = vectorizer.get_feature_names_out()
        features = plot_feature_effects(50)
        all_feature_names[0].update(features[0])
        all_feature_names[1].update(features[1])
    print(accs, f1, rocauc)
    print(all_feature_names[0])
    print(all_feature_names[1])
    with open("kmeans/kmeans_acc_conversations.pkl", "wb") as outfile:
        pickle.dump({"f1": f1, "accuracy": accs}, outfile)
