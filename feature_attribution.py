from preprocess_data import get_prism_convos, get_cad_convos
import argparse
import inseq
from data_selection_erasure import (
    evaluation_prism,
    evaluation_cad,
)
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import login

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
    parser.add_argument(
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    args = parser.parse_args()
    if args.token:
        login(args.token)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    model = inseq.load_model(
        args.model,
        "integrated_gradients",
        model_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
    )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_pickle(
        f"llama_beliefs/{args.model.split('/')[1]}_{args.dataset}_answers.gz"
    )
    if args.dataset == "prism":
        convo_func = get_prism_convos
        evaluation = evaluation_prism
    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        evaluation = evaluation_cad
        concept = "morality"

    df_questions = pd.read_pickle(f"{args.data_folder}/questions.gz")
    df_questions = df_questions.loc[
        df_questions["q_id"].isin([f"q_{i}" for i in range(50)])
    ]
    questions_answers = dict(
        zip(df_questions.q_id, df_questions.correct_answer)
    )
    question = df_questions[df_questions["q_id"] == "q_33"].iloc[0]["question"]

    for c in [f"q_{i}" for i in range(50)]:
        df[c] = df[c].str.lower() == questions_answers[c]

    df["accuracy"] = df[[f"q_{i}" for i in range(50)]].mean(axis=1) * 100

    c_ids = df[
        df["conversation_id"].isin(
            [
                c
                for item in evaluation[concept]
                for c in evaluation[concept][item]
            ]
        )
    ][["conversation_id", "accuracy"]].sort_values(by="accuracy")
    c_id_low = c_ids.iloc[0]["conversation_id"]
    c_id_high = c_ids.iloc[-1]["conversation_id"]

    df_convo = df[df["conversation_id"] == c_id_low]
    convo = convo_func(df_convo)
    tokenized_convo = tokenizer.apply_chat_template(
        convo[0] + [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = model.attribute(
        tokenized_convo,
        generation_args={
            "max_new_tokens": 100,
            "do_sample": False,
        },
        n_steps=100,
        internal_batch_size=4,
    )
    html = out.show(return_html=True)
    with open(
        f"{args.results_folder}/feature_attrb_{args.dataset}_{concept}_low_acc.html",
        "w",
    ) as f:
        f.write(html)

    df_convo = df[df["conversation_id"] == c_id_high]
    convo = convo_func(df_convo)
    tokenized_convo = tokenizer.apply_chat_template(
        convo[0] + [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = model.attribute(
        tokenized_convo,
        generation_args={
            "max_new_tokens": 100,
            "do_sample": False,
        },
        n_steps=100,
        internal_batch_size=4,
    )
    html = out.show(return_html=True)
    with open(
        f"{args.results_folder}/feature_attrb_{args.dataset}_{concept}_high_acc.html",
        "w",
    ) as f:
        f.write(html)
