import argparse
import json
import pandas as pd
import pickle
from huggingface_hub import login
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocess_data import get_prism_convos

from together import Together

client = Together()

# prompt from https://arxiv.org/abs/2508.15830

prompt = """You will be given a conversation between a user and an LLM assistant in text form. Your task is to infer the user’s demograpics by selecting one option per category below and return the result in the exact JSON schema.

Categories:
Age: {{18-34 years old, 35-54 years old, 55+ years old}}
Gender: {{Male, Female, Non-binary}}
English proficiency: {{Native speaker, Non-Native speaker}}
Educational Background: {{Low, Middle, High}}
Marital Status: {{Never been married, Married, Divorced, Widowed}}
Race/Ethnicity: {{Asian, Black, Hispanic, White}}
Religion: {{No Affiliation, Christian, Jewish, Muslim}}

Final answer:
{{
"Age": ,
"Gender": ,
"English proficiency": ,
"Educational Background": ,
"Marital Status": ,
"Race/Ethnicity": ,
"Religion": ,
}}

Conversation: "{conversation}"
Final answer:"""


def predict(convo: str) -> str:
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2.6",
        messages=[
            {"role": "user", "content": prompt.format(conversation=convo)}
        ],
        reasoning={"enabled": False},
    )

    return response.choices[0].message.content


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
        "-rf",
        "--results_folder",
        type=str,
        default="",
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
    df = pd.read_pickle(f"data/prism_preprocessed.gz")
    convo_func = get_prism_convos
    convos = convo_func(df)
    convos = [
        "\n".join([t["role"] + ": " + t["content"] for t in convo])
        for convo in convos
    ]

    lines = [
        {
            "custom_id": f"request-{i}",
            "body": {
                "model": "moonshotai/Kimi-K2.6",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt.format(conversation=convo),
                    }
                ],
                "reasoning": {"enabled": False},
            },
        }
        for i, convo in enumerate(convos)
    ]
    with open("batch_input.jsonl", "w") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")
    file_resp = client.files.upload(
        file="batch_input.jsonl", purpose="batch-api", check=False
    )
    print(file_resp)
    file_id = file_resp.id

    batch = client.batches.create(
        input_file_id=file_id, endpoint="/v1/chat/completions"
    )

    print(batch.job.id)
