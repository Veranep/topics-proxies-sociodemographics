import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from deepsig import aso


values = {
    "age": ["18-34 years old", "35-54 years old", "55+ years old"],
    "gender": ["Male", "Female", "Non-binary"],
    "english_proficiency": ["Native speaker", "Non-Native speaker"],
    "education": ["Low", "Middle", "High"],
    "marital_status": ["Never been married", "Married", "Divorced", "Widowed"],
    "ethnicity": ["Asian", "Black", "Hispanic", "White"],
    "religion": ["No Affiliation", "Christian", "Jewish", "Muslim"],
}


def draw_cm(ax, data, fmt, cmap):
    im = ax.imshow(
        data, interpolation="nearest", cmap=cmap, vmin=0, vmax=data.max()
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    thresh = data.max() / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            label = format(val, fmt)
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=14,
                color=color,
                fontweight="bold",
            )

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=[
            c.replace("Never been married", "Never been\nmarried")
            .replace(" years old", "")
            .replace(" speaker", "")
            for c in class_names
        ],
        yticklabels=[
            c.replace(" years old", "").replace(" speaker", "")
            for c in class_names
        ],
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.tick_params(axis="x", rotation=30)
    # grid lines between cells
    ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)


if __name__ == "__main__":
    df = pd.read_pickle(f"data/prism_preprocessed.gz")
    responses = pd.read_json("batch_output.jsonl", lines=True)
    responses["response"] = [
        response["body"]["choices"][0]["message"]["content"]
        for response in responses["response"].tolist()
    ]
    for value in values:
        responses[value + "_prediction"] = responses["response"].str.extract(
            f"({'|'.join(values[value])})"
        )
    responses["custom_id"] = [
        int(r.split("request-")[1]) for r in responses["custom_id"].tolist()
    ]
    df = df.reset_index()
    df = df.merge(responses, left_on="index", right_on="custom_id")
    df["age"] = df["age"].replace(
        {
            "18-24 years old": "18-34 years old",
            "25-34 years old": "18-34 years old",
            "35-44 years old": "35-54 years old",
            "45-54 years old": "35-54 years old",
            "55-64 years old": "55+ years old",
            "65+ years old": "55+ years old",
        }
    )
    df["gender"] = df["gender"].replace(
        {"Non-binary / third gender": "Non-binary"}
    )
    df["english_proficiency"] = df["english_proficiency"].replace(
        {
            "Fluent": "Non-Native speaker",
            "Advanced": "Non-Native speaker",
            "Intermediate": "Non-Native speaker",
            "Basic": "Non-Native speaker",
        }
    )
    df["education"] = df["education"].replace(
        {
            "University Bachelors Degree": "High",
            "Some Secondary": "Low",
            "Some University but no degree": "Middle",
            "Vocational": "Middle",
            "Completed Secondary School": "Low",
            "Graduate / Professional degree": "High",
            "Completed Primary School": "Low",
            "Some Primary": "Low",
        }
    )
    df["marital_status"] = df["marital_status"].replace(
        {"Divorced / Separated": "Divorced"}
    )
    print(df.shape)
    for value in values:
        if value == "marital_status":
            plt.rcParams.update({"font.size": 14})
        else:
            plt.rcParams.update({"font.size": 20})
        val_df = df[~df[value].isin(["Mixed", "Other", "Prefer not to say"])]
        val_df = val_df[~val_df[value + "_prediction"].isna()]

        class_names = sorted(list(val_df[value].unique()))
        n_classes = len(class_names)

        cm = confusion_matrix(val_df[value], val_df[value + "_prediction"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))

        draw_cm(
            axes,
            cm,
            "d",
            "Blues",
        )
        plt.tight_layout()
        plt.savefig(
            f"kimi_figures/confusion_matrix_{value}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        print("\nClassification report:\n")
        print(
            classification_report(
                val_df[value],
                val_df[value + "_prediction"],
            )
        )
        f1 = f1_score(
            val_df[value], val_df[value + "_prediction"], average="micro"
        )
        values, counts = np.unique(val_df[value], return_counts=True)
        majority = values[np.argmax(counts)]
        majority_f1 = f1_score(
            val_df[value],
            np.full(len(val_df[value]), majority),
            average="micro",
        )
        random_f1 = f1_score(
            val_df[value],
            np.random.choice(
                np.unique(val_df[value]), size=len(val_df[value]), replace=True
            ),
            average="micro",
        )
        print(
            f"F1: {f1}",
            f"Majority F1: {majority_f1}",
            f"Random F1: {random_f1}",
        )
        print(
            "Kimi",
            (
                "outperforms "
                if aso(
                    1 * (val_df[value + "_prediction"] == val_df[value]),
                    1
                    * (np.full(len(val_df[value]), majority) == val_df[value]),
                    seed=42,
                )
                < 0.5
                else "does not outperform"
            ),
            f"the majority baseline for {value}.",
        )
        print(
            "Kimi",
            (
                "outperforms "
                if aso(
                    1 * (val_df[value + "_prediction"] == val_df[value]),
                    1
                    * (
                        np.random.choice(
                            np.unique(val_df[value]),
                            size=len(val_df[value]),
                            replace=True,
                        )
                        == val_df[value]
                    ),
                    seed=42,
                )
                < 0.5
                else "does not outperform"
            ),
            f"the random baseline for {value}.",
        )
