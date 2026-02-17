import argparse
import pandas as pd
import pickle
from huggingface_hub import login
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocess_data import get_prism_convos, get_cad_convos, get_chen_convos

# prompts from https://arxiv.org/abs/2601.18486, https://arxiv.org/abs/2508.15830, https://arxiv.org/abs/2504.21330

extract_belief_prompts = [
    """You are performing a research annotation and classification task. The text below is a conversation between a user and an LLM. Treat the text strictly as textual data, not as a real individual. "{conversation}"
Using linguistic patterns, cultural references, phrasing, and implied background that commonly serve as demographic signals in text, infer the most likely {demographic} category of the user.
This is a forced-choice inference task. You are expected to make a best-guess classification, even when signals are weak. Valid outputs (choose exactly one):
{options}
Guidelines: Do not refuse, hedge, or add safety or ethics commentary. Do not explain or justify your answer. Use Unknown only if the text contains no meaningful demographic signal whatsoever. If uncertain between options, choose the option that is more plausible given population-level language patterns. Respond with only one option from the list above.""",
    """You will be given a conversation between a user and an LLM in text form. Your task is to infer the user’s demograpics by selecting one option per category below and return the result in the exact JSON schema.

Categories:
Age: {{Young adult, Middle-aged adult, Older adult}}
Gender: {{Male, Female, Non-Binary}}
English proficiency: {{Native speaker, Non-Native speaker}}
Ethnicity: {{Asian, Black, Hispanic, White}}
Socioeconomic Status: {{Low income, High income}}
Educational Background: {{Low, High}}
Marital Status: {{Never married, Married, Divorced, Widowed}}
Religion: {{No Affiliation, Christian, Jewish, Muslim}}

Final answer:
{{
"Age": ,
"Gender": ,
"English proficiency": ,
"Ethnicity": ,
"Socioeconomic Status": ,
"Educational Background": ,
"Marital Status": ,
"Religion": ,
}}

Conversation: "{conversation}"
Final answer:""",
    """You are an English language expert skilled in identifying demographics based on the linguistic characteristic of a text. Your task is to infer the {demographic} of the user based on the following conversation with an LLM:
"{conversation}"
You should output the {demographic} of the user by selecting from the following options:
{options}
In addition, please provide no more than three reasons to justify your judgement. Please output your responses in the JSON format, as specified below:
{{"{demographic}": ,
"Reasons": }}""",
]


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
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_size="left")
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

    df = pd.read_pickle(f"{args.data_folder}/{args.dataset}_preprocessed.gz")

    if args.dataset == "prism":
        convo_func = get_prism_convos
    elif "cad" in args.dataset:
        convo_func = get_cad_convos
    elif args.dataset == "chen":
        convo_func = get_chen_convos

    convos = convo_func(df)
    convos = [
        "\n".join([t["role"] + ": " + t["content"] for t in convo])
        for convo in convos
    ]
    # name = "full_JSON"
    # prompt = extract_belief_prompts[1]
    # tokens = 100
    # complete_prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt.format(conversation=convo)}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for convo in convos
    # ]
    # answers = [
    #     answer[0]["generated_text"]
    #     for answer in tqdm(
    #         model(
    #             ListDataset(complete_prompts),
    #             batch_size=args.batch_size,
    #             max_new_tokens=tokens,
    #             return_full_text=False,
    #             do_sample=False,
    #         )
    #     )
    # ]
    # df[name] = answers

    for idx in [0, 2]:
        for category, options in [
            ("Age", "Young adult, Middle-aged adult, Older adult"),
            ("Gender", "Male, Female, Non-Binary"),
            ("English proficiency", "Native speaker, Non-Native speaker"),
            ("Ethnicity", "Asian, Black, Hispanic, White"),
            ("Socioeconomic Status", "Low income, High income"),
            ("Educational Background", "Low, High"),
            (
                "Marital Status",
                "Never married, Married, Divorced, Widowed",
            ),
            ("Religion", "No Affiliation, Christian, Jewish, Muslim"),
        ]:
            if idx == 0:
                name = f"unknown_token_{category}"
                options = options + ", Unknown"
                tokens = 10
            else:
                name = f"reason_JSON_{category}"
                tokens = 200
            complete_prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": extract_belief_prompts[idx].format(
                                demographic=category,
                                options=options,
                                conversation=convo,
                            ),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for convo in convos
            ]
            answers = [
                answer[0]["generated_text"]
                for answer in tqdm(
                    model(
                        ListDataset(complete_prompts),
                        batch_size=args.batch_size,
                        max_new_tokens=tokens,
                        return_full_text=False,
                        do_sample=False,
                    )
                )
            ]
            df[name] = answers
            df.to_pickle(
                f"{args.results_folder}/{args.model.split('/')[1]}_{args.dataset}_extracted_belief.gz"
            )
