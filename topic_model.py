import argparse
import hdbscan
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap


def extract_top_words_per_cluster(tf_idf, count, texts_by_cluster, top_n):
    words = count.get_feature_names_out()
    labels = list(texts_by_cluster["cluster_id"])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -top_n:]
    top_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }

    return top_words


def c_tf_idf(texts, m, ngram_range=(1, 2)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        texts
    )
    t = count.transform(texts).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    clusters = {
        -1: "Outliers",
        0: "Finance",
        1: "Write thank you",
        2: "UX Design",
        3: "Fashion",
        4: "School",
        5: "Food trip",
        6: "Tourism",
        7: "Charity",
        8: "Travel tips",
        9: "Multilingual company skills",
        10: "Illness support",
        11: "Urgent emails",
        12: "Cafes",
        13: "Marketing",
        14: "Wine, cuisine",
        15: "Village trip",
    }
    if not clusters:
        df = pd.read_pickle(f"{args.data_folder}/cad_en_preprocessed.gz")
        print("Loaded data")
        sentences = df["first_turn_prompt"].tolist()
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = model.encode(sentences, show_progress_bar=True)
        print("Got embeddings")
        umap_embeddings = umap.UMAP(
            random_state=42,
            n_neighbors=15,
            n_components=20,
            min_dist=0.0,
            metric="cosine",
        ).fit_transform(embeddings)
        print("Umapped")
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=80,
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit(umap_embeddings)
        print(
            f"  Created {len(set(cluster.labels_))-1} clusters with a minimum size of 265 texts with HDBScan."
        )
        df["cluster_id"] = cluster.labels_
        cluster_dict = dict(
            zip(df["first_turn_prompt"].tolist(), df["cluster_id"].tolist())
        )
        texts_by_cluster = df.groupby(["cluster_id"], as_index=False).agg(
            {"first_turn_prompt": " ".join}
        )

        # Run tf-idf, then use that to identify top 20 uni/bigrams for each cluster
        tf_idf, count = c_tf_idf(
            texts_by_cluster.first_turn_prompt.values, m=len(df)
        )
        top_words = extract_top_words_per_cluster(
            tf_idf, count, texts_by_cluster, top_n=20
        )
        print(top_words)

        utterances = pd.read_pickle(
            f"{args.data_folder}/cad_en_utterances_preprocessed.gz"
        )
        utterances["topic"] = utterances["first_turn_prompt"].map(cluster_dict)
    else:
        utterances["topic"] = utterances["topic"].map(clusters)

    utterances.to_pickle(
        f"{args.data_folder}/cad_en_utterances_preprocessed.gz"
    )
