import argparse
from data_selection_erasure import (
    neurons_prism,
    neurons_cad,
    evaluation_prism,
    evaluation_cad,
)

from editor import Concept, unlearn_concept
from evals import TransformerLensModel
import numpy as np
import pandas as pd
import pickle
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from preprocess_data import get_prism_convos, get_cad_convos


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

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_pickle(f"{args.data_folder}/{args.dataset}_preprocessed.gz")

    if args.dataset == "prism":
        convo_func = get_prism_convos
        neurons = neurons_prism
        evaluation = evaluation_prism
    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        neurons = neurons_cad
        evaluation = evaluation_cad

    df_questions = pd.read_pickle(f"{args.data_folder}/questions.gz")
    # df_questions = df_questions.loc[
    #     df_questions["q_id"].isin([f"q_{i}" for i in range(50)])
    # ]
    df_questions = df_questions.loc[df_questions["q_id"] == "q_0"]
    for item in evaluation:
        # eval_convos = [
        #     c_id
        #     for group in evaluation[item]
        #     for c_id in evaluation[item][group]
        # ]
        eval_convos = [
            "c193",
            "c204",
            "c229",
            "c900",
            "c241",
            "c266",
            "c7688",
            "c5969",
        ]
        evaluation_df = df.loc[df["conversation_id"].isin(eval_convos)]
        convos = convo_func(evaluation_df)
        if item in neurons:
            for c in neurons[item]:
                features = neurons[item][c]
                concept = Concept(name=c, k=0.4, value=36, features=features)
                model = HookedTransformer.from_pretrained(
                    args.model,
                    device=device,
                )
                tm = TransformerLensModel(model)
                result_df = evaluation_df.reset_index(drop=True)

                with unlearn_concept(model, concept):
                    for row in tqdm(df_questions.itertuples(index=False)):
                        convos_and_questions = [
                            tokenizer.apply_chat_template(
                                convo
                                + [{"role": "user", "content": row.question}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for convo in convos
                        ]
                        tokens = 1
                        outputs = tm.generate_multiple(
                            convos_and_questions,
                            batch_size=4,
                            max_new_tokens=tokens,
                            do_sample=False,
                            verbose=True,
                        )

                        # outputs = [
                        #     o.split("<|end_header_id|>")[-1].strip()
                        #     for o in tm.generate_multiple(
                        #         convos_and_questions,
                        #         batch_size=4,
                        #         max_new_tokens=tokens,
                        #         do_sample=False,
                        #         verbose=True,
                        #     )
                        # ]

                        result_df = pd.concat(
                            [result_df, pd.DataFrame({row.q_id: outputs})],
                            axis=1,
                        )
                        result_df.to_pickle(
                            f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_pisces_{item}_{c}_check_answers.gz"
                        )
