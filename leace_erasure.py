import argparse
from data_selection_erasure import (
    leace_convos_prism,
    leace_convos_cad,
    evaluation_prism,
    evaluation_cad,
)
import datasets
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llama import scrub_llama

from preprocess_data import get_prism_convos, get_cad_convos


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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

    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_pickle(f"{args.data_folder}/{args.dataset}_preprocessed.gz")

    if args.dataset == "prism":
        convo_func = get_prism_convos
        leace_convos = leace_convos_prism
        leace_df = pd.read_pickle(f"{args.data_folder}/cad_en_preprocessed.gz")
        leace_convo_func = get_cad_convos
        evaluation = evaluation_prism
    elif "cad" in args.dataset:
        convo_func = get_cad_convos
        leace_convos = leace_convos_cad
        leace_df = pd.read_pickle(f"{args.data_folder}/prism_preprocessed.gz")
        leace_convo_func = get_prism_convos
        evaluation = evaluation_cad

    convos = convo_func(df)
    df_questions = pd.read_pickle(f"{args.data_folder}/questions.gz")
    df_questions = df_questions.loc[
        df_questions["q_id"].isin([f"q_{i}" for i in range(50)])
    ]
    for item in evaluation:
        eval_convos = [
            c_id
            for group in evaluation[item]
            for c_id in evaluation[item][group]
        ]
        evaluation_df = df.loc[df["conversation_id"].isin(eval_convos)]

        if item in leace_convos:
            leace_cs = [
                c_id
                for group in leace_convos[item]
                for c_id in leace_convos[item][group]
            ]
            selected_leace_df = leace_df.loc[
                leace_df["conversation_id"].isin(leace_cs)
            ]
            reverse_label_dict = {
                c_id: group
                for group in leace_convos[item]
                for c_id in leace_convos[item][group]
            }
            leace_labels = selected_leace_df["conversation_id"].map(
                reverse_label_dict
            )
            leace_cs = leace_convo_func(selected_leace_df)
            leace_cs = [
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                )
                | {"label": leace_labels.iloc[i]}
                for i, convo in enumerate(leace_cs)
            ]
            leace_dataset = datasets.Dataset.from_pandas(
                pd.DataFrame(leace_cs)
            )
            leace_dataset = leace_dataset.class_encode_column("label")
            scrubber = scrub_llama(model, leace_dataset, z_column="label")
            with scrubber.scrub(model):
                leace_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
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
                    outputs = [
                        answer[0]["generated_text"]
                        for answer in tqdm(
                            leace_pipeline(
                                ListDataset(convos_and_questions),
                                batch_size=8,
                                max_new_tokens=tokens,
                                return_full_text=False,
                                do_sample=False,
                            )
                        )
                    ]

                    evaluation_df = pd.concat(
                        [evaluation_df, pd.DataFrame({row.q_id: outputs})],
                        axis=1,
                    )
                    evaluation_df.to_pickle(
                        f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_leace_{item}_answers.gz"
                    )
