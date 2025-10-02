import itertools
import argparse
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import os
import pandas as pd
import itertools


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
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    args = parser.parse_args()
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
    model = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not model.tokenizer.pad_token_id:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    df = pd.read_pickle(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_preprocessed.gz",
        compression="gzip",
    )

    if os.path.isfile(
        "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions.gz"
    ):
        df = pd.read_pickle(
            "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions.gz",
            compression="gzip",
        )
    else:
        evals = {
            "medical": list(
                set(
                    pd.read_csv("old/medical_llama_prompts.csv")[
                        "prompts"
                    ].tolist()
                    + pd.read_csv("old/medical_qwen_prompts.csv")[
                        "prompts"
                    ].tolist(),
                )
            )
        }
        questions = list(itertools.chain.from_iterable(evals.values())) * len(
            df
        )
        evaluation = list(
            itertools.chain.from_iterable(
                [[ev] * len(evals[ev]) for ev in evals]
            )
        ) * len(df)

        df = pd.concat(
            [df] * len(list(itertools.chain.from_iterable(evals.values()))),
            ignore_index=True,
        )

        df["evaluation"] = evaluation
        df["questions"] = questions
        df.to_pickle(
            "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions.gz"
        )

    conversations_with_questions = [
        [
            {"role": turn["role"], "content": turn["content"]}
            for turn in df.iloc[i]["conversation_history"]
            if turn["role"] == "user" or turn["if_chosen"] == True
        ]
        + [{"role": "user", "content": df.iloc[i]["questions"]}]
        for i in range(len(df))
    ]
    print(conversations_with_questions[0])
    answers = [
        answer[0]["generated_text"].lower()
        for answer in tqdm(
            model(
                ListDataset(conversations_with_questions),
                batch_size=args.batch_size,
                do_sample=False,
                max_new_tokens=1,
                return_full_text=False,
            ),
            total=len(conversations_with_questions),
        )
    ]
    df["answers"] = answers
    df.to_pickle(
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_answers"
    )
