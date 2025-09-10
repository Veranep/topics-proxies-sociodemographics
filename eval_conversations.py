import argparse
from itertools import permutations, product
import numpy as np
import pandas as pd
from utils import explicit_indicators, templates


def get_conversations(
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


def ask_questions(convos, questions):
    convo_len = len(convos[0])
    current_convos = [[] for _ in range(len(convos))]
    user_turns = [
        [{"role": "user", "content": turn} for turn in convo]
        for convo in convos
    ]
    question_turns = [[{"role": "user", "content": q}] for q in questions]
    # for i in range(convo_len):
    #     for j, current_convo in enumerate(current_convos):
    #         current_convo.append(user_turns[j][i])
    #     current_convos = [
    #         convo[0]["generated_text"]
    #         for convo in tqdm(
    #             model(
    #                 current_convos,
    #                 batch_size=batch_size,
    #                 do_sample=False,
    #                 max_new_tokens=100,
    #             ),
    #             total=len(current_convos),
    #         )
    #     ]
    print(current_convos[0], question_turns[0])
    convo_with_questions = list(
        map(
            lambda x: x[0] + x[1],
            list(product(current_convos, question_turns)),
        )
    )
    print(convo_with_questions)


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
    parser.add_argument(
        "-d",
        "--demo",
        type=str,
        default="gender",
        help="Demographic group to evaluate",
    )
    args = parser.parse_args()
    np.random.seed(42)
    stereotypes = pd.read_csv("stereotypes - Updated.csv").drop(
        columns=["source"]
    )
    initial_conversations = {
        "neutral": get_conversations(
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
            ) = get_conversations(
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
                "Hi, I am {0}.".format(indicator)
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
    for e in evals:
        questions = evals[e]
        ask_questions(initial_conversations["neutral"], questions)
