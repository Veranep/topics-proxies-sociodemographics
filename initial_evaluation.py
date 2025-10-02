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
import xml.etree.ElementTree as ET


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
        all_questions = {
            "health_misinfo": [
                topic.find("question").text
                + " Respond with either 'Yes' or 'No' and no additonal text."
                for topic in ET.parse("data/misinfo-2022-topics.xml")
                .getroot()
                .findall("topic")
            ],
        }
        all_gold_answers = {
            "health_misinfo": [
                topic.find("answer").text
                for topic in ET.parse("data/misinfo-2022-topics.xml")
                .getroot()
                .findall("topic")
            ],
        }
        questions = [
            q
            for q in list(
                itertools.chain.from_iterable(all_questions.values())
            )
            for _ in range(len(df))
        ]
        gold_answers = [
            q
            for q in list(
                itertools.chain.from_iterable(all_gold_answers.values())
            )
            for _ in range(len(df))
        ]
        evaluation = [
            e
            for e in list(
                itertools.chain.from_iterable(
                    [[ev] * len(all_questions[ev]) for ev in all_questions]
                )
            )
            for _ in range(len(df))
        ]

        df = pd.concat(
            [df]
            * len(list(itertools.chain.from_iterable(all_questions.values()))),
            ignore_index=True,
        )

        df["evaluation"] = evaluation
        df["question"] = questions
        df["gold_answer"] = gold_answers
        df.to_pickle(
            "/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions.gz"
        )

    conversations_with_questions = [
        tokenizer.apply_chat_template(
            [
                {"role": turn["role"], "content": turn["content"]}
                for turn in df.iloc[i]["conversation_history"]
                if turn["role"] == "user" or turn["if_chosen"] == True
            ]
            + [{"role": "user", "content": df.iloc[i]["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(len(df))
    ]
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
    df["answer"] = answers
    df.to_pickle(
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_answers.gz"
    )
