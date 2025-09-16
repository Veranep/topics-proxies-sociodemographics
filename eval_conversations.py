import argparse
from itertools import permutations, product
import numpy as np
import pandas as pd
from utils import explicit_indicators, templates
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_user_turns(
    demographic, value, stereotypes, indicators, templates, n, neutral
):
    if neutral:
        neutral_items = stereotypes[stereotypes["dem_attribute"] == "neutral"][
            ["type", "value1", "value2"]
        ].values
        np.random.shuffle(neutral_items)
        neutral_convos = [
            [
                np.random.choice(templates[item[0]]).format(item[1], item[2])
                for item in comb
            ]
            for comb in permutations(neutral_items, 6)
        ][:n]
        return neutral_convos
    else:
        stereo_convos = []
        explicit_stereo_convos = []
        stereo_items = stereotypes[
            stereotypes["dem_attribute"] == demographic
        ][stereotypes["dem_value"] == value][
            ["type", "value1", "value2"]
        ].values
        if len(stereo_items):
            stereo_convos = [
                [
                    np.random.choice(templates[item[0]]).format(
                        item[1], item[2]
                    )
                    for item in stereo_items[
                        np.random.choice(len(stereo_items), 6, replace=False)
                    ]
                ]
                for _ in range(n)
            ]
            explicit_stereo_convos = [
                [
                    "Hi, I am {0}.".format(
                        str(np.random.choice(indicators, 1)[0])
                    )
                ]
                + stereo_convos[i]
                for i in range(n)
            ]
        return stereo_convos, explicit_stereo_convos


def get_conversations(convos, model, batch_size):
    convo_len = len(convos[0])
    current_convos = [[] for _ in range(len(convos))]
    user_turns = [
        [{"role": "user", "content": turn} for turn in convo]
        for convo in convos
    ]
    for i in range(convo_len):
        for j, current_convo in enumerate(current_convos):
            current_convo.append(user_turns[j][i])
        current_convos = [
            convo[0]["generated_text"]
            for convo in tqdm(
                model(
                    current_convos,
                    batch_size=batch_size,
                    do_sample=False,
                    max_new_tokens=250,
                ),
                total=len(current_convos),
            )
        ]
    return current_convos


def ask_questions(convos, questions, model, batch_size):
    question_turns = [[{"role": "user", "content": q}] for q in questions]
    if convos:
        convos_with_questions = list(
            map(
                lambda x: x[0] + x[1],
                list(product(convos, question_turns)),
            )
        )
    else:
        convos_with_questions = question_turns
    print(convos_with_questions[0])
    answers = [
        convo[0]["generated_text"]
        for convo in tqdm(
            model(
                convos_with_questions,
                batch_size=batch_size,
                do_sample=False,
                max_new_tokens=100,
            ),
            total=len(convos_with_questions),
        )
    ]
    all_questions = [c[-1]["content"] for c in convos_with_questions]
    if convos:
        convo_ids = np.repeat(range(len(convos)), len(questions))
    else:
        convo_ids = []
    return all_questions, answers, convo_ids


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
        "-n",
        type=int,
        default=None,
        help="Number of samples",
    )
    # parser.add_argument(
    #     "-d",
    #     "--demo",
    #     type=str,
    #     default="gender",
    #     help="Demographic group to evaluate",
    # )
    args = parser.parse_args()
    np.random.seed(42)
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
    stereotypes = pd.read_csv("stereotypes - Updated.csv").drop(
        columns=["source"]
    )
    initial_conversations = {
        "neutral": get_user_turns(
            None, None, stereotypes, None, templates, n=args.n, neutral=True
        ),
    }
    for demographic in explicit_indicators:
        initial_conversations[demographic] = {}
        for value in explicit_indicators[demographic]:
            initial_conversations[demographic][value] = {}
            (
                initial_conversations[demographic][value]["stereo"],
                initial_conversations[demographic][value]["explicit_stereo"],
            ) = get_user_turns(
                demographic,
                value,
                stereotypes,
                [
                    indicator
                    for v in explicit_indicators[demographic]
                    if v != value
                    for indicator in explicit_indicators[demographic][v]
                ],
                templates,
                n=args.n,
                neutral=False,
            )
            initial_conversations[demographic][value]["explicit"] = [
                ["Hi, I am {0}.".format(indicator)]
                for indicator in explicit_indicators[demographic][value]
            ]
    clean_medical_data = lambda x: x.replace(
        " Respond with either 'Yes' or 'No' and no additonal text.",
        "",
    )
    evals = {
        "medical": list(
            set(
                list(
                    map(
                        clean_medical_data,
                        pd.read_csv("medical_llama_prompts.csv")[
                            "prompts"
                        ].tolist(),
                    )
                )
                + list(
                    map(
                        clean_medical_data,
                        pd.read_csv("medical_qwen_prompts.csv")[
                            "prompts"
                        ].tolist(),
                    )
                )
            )
        )
    }
    conversations_dict = {
        "demographic": [],
        "value": [],
        "setting": [],
        "conversations": [],
    }
    results_dict = {
        "eval": [],
        "demographic": [],
        "value": [],
        "setting": [],
        "questions": [],
        "answers": [],
        "convo_ids": [],
    }
    neutral_convos = get_conversations(
        initial_conversations["neutral"], model, args.batch_size
    )
    conversations_dict["demographic"].append(None)
    conversations_dict["value"].append(None)
    conversations_dict["setting"].append("neutral")
    conversations_dict["conversations"].append(neutral_convos)
    for e in evals:
        questions = evals[e]
        results_dict["eval"].append(e)
        results_dict["demographic"].append(None)
        results_dict["value"].append(None)
        results_dict["setting"].append("question_only")
        all_questions, answers, convo_ids = ask_questions(
            None, questions, model, args.batch_size
        )
        results_dict["questions"].append(all_questions)
        results_dict["answers"].append(answers)
        results_dict["convo_ids"].append(convo_ids)
        results_dict["eval"].append(e)
        results_dict["demographic"].append(None)
        results_dict["value"].append(None)
        results_dict["setting"].append("neutral")
        all_questions, answers, convo_ids = ask_questions(
            neutral_convos, questions, model, args.batch_size
        )
        results_dict["questions"].append(all_questions)
        results_dict["answers"].append(answers)
        results_dict["convo_ids"].append(convo_ids)
    results_df = pd.DataFrame(data=results_dict)
    results_df.to_pickle(
        f"/scratch/vneplen/implicit-personalization-stereotypes-model-responses/{args.model.split('/')[1]}_results.gz"
    )
    print(results_df, results_df.shape)
    conversations_df = pd.DataFrame(data=conversations_dict)
    conversations_df.to_pickle(
        f"/scratch/vneplen/implicit-personalization-stereotypes-model-responses/{args.model.split('/')[1]}_conversations.gz"
    )
    for demographic in initial_conversations:
        if demographic == "neutral":
            continue
        for value in initial_conversations[demographic]:
            for setting in initial_conversations[demographic][value]:
                if not initial_conversations[demographic][value][setting]:
                    print("no conversations for ", demographic, value, setting)
                    continue
                convos = get_conversations(
                    initial_conversations[demographic][value][setting],
                    model,
                    args.batch_size,
                )
                conversations_dict["demographic"].append(demographic)
                conversations_dict["value"].append(value)
                conversations_dict["setting"].append(setting)
                conversations_dict["conversations"].append(convos)
                for e in evals:
                    print(demographic, value, setting, e)
                    questions = evals[e]
                    results_dict["eval"].append(e)
                    results_dict["demographic"].append(demographic)
                    results_dict["value"].append(value)
                    results_dict["setting"].append(setting)
                    all_questions, answers, convo_ids = ask_questions(
                        convos, questions, model, args.batch_size
                    )
                    results_dict["questions"].append(all_questions)
                    results_dict["answers"].append(answers)
                    results_dict["convo_ids"].append(convo_ids)
        results_df = pd.DataFrame(data=results_dict)
        results_df.to_pickle(
            f"/scratch/vneplen/implicit-personalization-stereotypes-model-responses/{args.model.split('/')[1]}_results.gz"
        )
        print(results_df, results_df.shape)
        conversations_df = pd.DataFrame(data=conversations_dict)
        conversations_df.to_pickle(
            f"/scratch/vneplen/implicit-personalization-stereotypes-model-responses/{args.model.split('/')[1]}_conversations.gz"
        )
