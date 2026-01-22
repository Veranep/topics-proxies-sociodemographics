import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from probing_new import get_convo

np.random.seed(42)


def k_means(df, model, layer, add_prompt, add_question):
    convos = [get_convo(df.iloc[i], test=add_question) for i in range(len(df))]
    for convo in convos:
        to_remove = []
        for i in range(len(convo)):
            if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for idx in to_remove:
            del convo[idx]
    if tokenizer.chat_template:
        inputs = [
            (
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        convo,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    + f" I think the {demographic} of this user is ",
                    return_tensors="pt",
                )
                if add_prompt
                else tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            )
            for convo in convos
        ]
    else:
        inputs = [tokenizer(inp, return_tensors="pt") for inp in convos]

    representations = [
        (
            [
                rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"][layer]
            ]
            if tokenizer.chat_template
            else [
                rep[-1, -1, :].detach().cpu().clone().to(torch.float)
                for rep in model(
                    **inp.to(device),
                    do_sample=False,
                    output_hidden_states=True,
                    max_new_tokens=1,
                    return_dict=True,
                )["hidden_states"][layer]
            ]
        )
        for inp in tqdm(inputs)
    ]
    results = {}
    for k in [2, 3, 4, 5, 6]:
        cluster_labels = KMeans(
            n_clusters=k, random_state=42, n_init="auto"
        ).fit_predict(representations)
        print(k, silhouette_score(representations, cluster_labels))
        results[k] = cluster_labels
    return results


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
        "-l",
        "--layer",
        type=int,
        default=None,
        help="Model layer to evaluate",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Whether to add an introspective sentence to the prompt",
    )
    parser.add_argument(
        "--question",
        action="store_true",
        help="Whether to add a question to the prompt",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template and not args.first:
        tokenizer.chat_template = chat_templates[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    health_misinfo = pd.read_pickle(
        "data/prism_questions_health_misinfo.gz",
        compression="gzip",
    )

    # with open("data/conv_ids_prism.pkl", "rb") as infile:
    #     conv_ids = pickle.load(infile)
    with open("data/q_ids_prism.pkl", "rb") as infile:
        q_ids = pickle.load(infile)

    if args.question:
        q_id_counts = Counter(
            [
                q
                for demo in q_ids[model_name]
                for q in q_ids[model_name][demo]
                if q in health_misinfo["question"].unique()
            ]
        )
        max_q_id = max(q_id_counts, key=q_id_counts.get)
        health_misinfo = health_misinfo.loc[
            health_misinfo["question"] == max_q_id
        ]

    health_misinfo = (
        health_misinfo.groupby(["conversation_id"]).first().reset_index()
    )

    clusters = k_means(
        health_misinfo, model, args.layer, args.prompt, args.question
    )
    with open(
        args.folder
        + f"/{args.model.split('/')[1]}_{args.layer}{'_prompt' if args.prompt else ''}{'_question' if args.question else ''}_kmeans_results.pkl",
        "wb",
    ) as outfile:
        pickle.dump(clusters, outfile)
