import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier


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
        feature_names[np.argsort(average_feature_effects)[n:][::-1]],
        feature_names[np.argsort(average_feature_effects)[n]],
    )


if __name__ == "__main__":
    conv_hist_df = pd.read_pickle(
        "prism_preprocessed.gz",
        compression="gzip",
    )[["conversation_id", "conversation_history"] + demographics]
    with open("data/conv_ids_prism.pkl", "rb") as infile:
        conv_ids = pickle.load(infile)
    for model in ["OLMo2", "OLMo3", "Llama", "Gemma"]:
        for demographic in demographics:
            if len(conv_ids[model][demographic]) == 0:
                continue
            selected_df = get_binary_subset(
                conv_hist_df[
                    conv_hist_df["conversation_id"].isin(
                        conv_ids[model][demographic]
                    )
                ],
                demographic,
            )
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                max_df=0.5,
                min_df=5,
                stop_words="english",
            )
            X_train = vectorizer.fit_transform(
                selected_df["conversation_history"].map(
                    lambda x: " ".join([t["content"] for t in x])
                )
            )
            # X_test = vectorizer.transform(data_test.data)
            clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
            clf.fit(X_train, selected_df[demographic].astype("int"))
            # pred = clf.predict(X_test)
            print(model, demographic)
            target_names = ["0", "1"]
            feature_names = vectorizer.get_feature_names_out()
            print(plot_feature_effects(demographic, 25))
