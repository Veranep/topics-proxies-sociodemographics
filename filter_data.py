import argparse
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

prompt = """Does the following text in any way contain mentions of {}? Answer just yes or no.
Text:{}"""  # from issuebench

prompt_options = {
    "hobbies": "hobbies, such as painting or playing tennis",
    "food": "food or drink items, such as fries or coffee",
    "traits": "character traits, such as being late or being bad at driving",
    "advice": "asking for advice, such as asking what to do in a specific situation",
    "recommendations": "asking for recommendations, such as for movies, books or travel destinations",
    "stereotypes": "stereotypical items or activities, such as make-up, going hunting, or eating fried chicken",
    "demographics": "demographics, such as gender, race or age",
}

filter_terms = [
    "I",
    "i",
    "my",
    "mine",
    "My",
    "Mine",
    "me",
    "Me",
    "me.",
    "me,",
    "me?",
    "mine.",
    "mine,",
    "mine?",
]


def get_first_user_turn(example):
    example["opening_prompt"] = example["conversation"][0]["content"]
    return example


def format_prompt(example, prompt_name):
    example[prompt_name] = [
        {
            "role": "user",
            "content": prompt.format(
                prompt_options[prompt_name], example["opening_prompt"]
            ),
        }
    ]
    return example


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
        default=None,
        help="Model",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "-token",
        type=str,
        default="",
        help="Huggingface token",
    )
    args = parser.parse_args()
    login(args.token)
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

    for dataset_name in [
        "shachardon/ShareLM",
        "lmsys/lmsys-chat-1m",
        "allenai/WildChat-4.8M",
    ]:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.map(get_first_user_turn)

        dataset = dataset.filter(
            lambda example: any(
                term in example["opening_prompt"].split()
                for term in filter_terms
            )
        )
        print(dataset.shape)

        for prompt_option in prompt_options:
            dataset = dataset.map(lambda x: format_prompt(x, prompt_option))
        pd_dataset = dataset.with_format("pandas")
        for prompt_option in prompt_options:
            questions = pd_dataset[prompt_option].tolist()
            answers = [
                "yes" in answer[0]["generated_text"][-1]["content"].lower()
                for answer in tqdm(
                    model(
                        ListDataset(questions),
                        batch_size=args.batch_size,
                        do_sample=False,
                        max_new_tokens=5,
                    ),
                    total=len(questions),
                )
            ]
            pd_dataset[prompt_option] = answers
        pd_dataset.to_pickle(
            f"scratch/vneplen/implicit-personalization-stereotypes-model-responses/{dataset_name.split('/')[1]}.gz"
        )
