import itertools
import argparse
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
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


def clean_fever_data(example):
    example["claim"] = (
        "Is it true that "
        + example["claim"].lower().strip(".")
        + "? Respond with either 'Yes' or 'No' and no additonal text."
    )
    return example


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
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="health_misinfo",
        help="Dataset to evaluate model on",
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
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions_{args.dataset}.gz"
    ):
        df = pd.read_pickle(
            f"/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions_{args.dataset}.gz",
            compression="gzip",
        )
    else:
        if args.dataset == "climate_fever":
            climate_fever = (
                load_dataset("tdiggelm/climate_fever", split="test")
                .shuffle(seed=42)
                .filter(
                    lambda x: x["claim_label"] == 0 or x["claim_label"] == 1
                )
                .select(list(range(50)))
            )
            questions = list(climate_fever.map(clean_fever_data)["claim"])
            answers = map(
                lambda x: 1 * (x == "no"), list(climate_fever["claim_label"])
            )

        elif args.dataset == "health_misinfo":
            questions = [
                topic.find("question").text
                + " Respond with either 'Yes' or 'No' and no additonal text."
                for topic in ET.parse("data/misinfo-2022-topics.xml")
                .getroot()
                .findall("topic")
            ]
            answers = [
                topic.find("answer").text
                for topic in ET.parse("data/misinfo-2022-topics.xml")
                .getroot()
                .findall("topic")
            ]
        elif args.dataset == "medical":
            questions = list(
                set(
                    pd.read_csv("old/medical_llama_prompts.csv")[
                        "prompts"
                    ].tolist()
                    + pd.read_csv("old/medical_qwen_prompts.csv")[
                        "prompts"
                    ].tolist(),
                )
            )
            answers = ["-"] * len(questions)
        print("preparing data")
        all_questions = [q for q in questions for _ in range(len(df))]
        gold_answers = [a for a in answers for _ in range(len(df))]
        evaluation = [args.dataset] * len(all_questions)

        print("extending df")

        df = pd.concat(
            [df] * len(questions),
            ignore_index=True,
        )

        df["evaluation"] = evaluation
        df["question"] = all_questions
        df["gold_answer"] = gold_answers
        print("got all data")
        df.to_pickle(
            f"/scratch/vneplen/sociodemographics-interpretability-mitigation/prism_questions_{args.dataset}.gz"
        )

    conversations_with_questions = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": turn["role"].replace("model", "assistant"),
                    "content": turn["content"],
                }
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
        f"/scratch/vneplen/sociodemographics-interpretability-mitigation/{args.model.split('/')[1]}_answers_climate_fever.gz"
    )
