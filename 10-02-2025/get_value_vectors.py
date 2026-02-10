import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens import utils

demographic_cols = [
    "age",
    "gender",
    "religion",
    "ethnicity",
    "employment_status",
    "education",
    "birth_region",
    "reside_region",
    "marital_status",  # not in facts paper
    "english_proficiency",  # not in facts paper
]

groups = {
    "age": [
        ("18-24 years old", 0),
        ("25-34 years old", 1),
        ("35-44 years old", 2),
        ("45-54 years old", 3),
        ("55-64 years old", 4),
        ("65+ years old", 5),
    ],
    "gender": [("Female", 0), ("Male", 1), ("Non-binary / third gender", 2)],
    "religion": [
        ("Christian", 0),
        ("Jewish", 1),
        ("Muslim", 2),
        ("No Affiliation", 3),
        ("Other", 4),
    ],
    "ethnicity": [
        ("Asian", 0),
        ("Black", 1),
        ("Hispanic", 2),
        ("Mixed", 3),
        ("Other", 4),
        ("White", 5),
    ],
    "employment_status": [
        ("Homemaker / Stay-at-home parent", 0),
        ("Retired", 1),
        ("Student", 2),
        ("Unemployed, not seeking work", 3),
        ("Unemployed, seeking work", 4),
        ("Working full-time", 5),
        ("Working part-time", 6),
    ],
    "education": [
        ("Completed Primary School", 0),
        ("Completed Secondary School", 1),
        ("Graduate / Professional degree", 2),
        ("Some Primary", 3),
        ("Some Secondary", 4),
        ("Some University but no degree", 5),
        ("University Bachelors Degree", 6),
        ("Vocational", 7),
    ],
    "birth_region": [
        ("Africa", 0),
        ("Americas", 1),
        ("Asia", 2),
        ("Europe", 3),
        ("Oceania", 4),
    ],
    "reside_region": [
        ("Africa", 0),
        ("Americas", 1),
        ("Asia", 2),
        ("Europe", 3),
        ("Oceania", 4),
    ],
    "marital_status": [
        ("Divorced / Separated", 0),
        ("Married", 1),
        ("Never been married", 2),
        ("Widowed", 3),
    ],
    "english_proficiency": [
        ("Advanced", 0),
        ("Basic", 1),
        ("Fluent", 2),
        ("Intermediate", 3),
        ("Native speaker", 4),
    ],
}

interesting_neurons = {
    "age": {
        "18-24 years old": [(15, 10570)],
        "25-34 years old": [],
        "35-44 years old": [(1, 4798), (9, 3315)],
        "45-54 years old": [(14, 3824), (13, 12242)],
        "55-64 years old": [(16, 9473)],
        "65+ years old": [(16, 9473), (15, 5335), (14, 3824)],
    },
    "gender": {
        "Female": [
            (15, 5337),
            (7, 6142),
            (13, 12372),
            (19, 13069),
            (9, 9353),
            (3, 10189),
            (6, 12345),
        ],
        "Male": [
            (15, 4128),
            (14, 4248),
        ],  # there are many more 'herself' neurons than 'himself' neurons
        "Non-binary / third gender": [],
    },
    "religion": {
        "Christian": [],
        "Jewish": [(25, 3), (12, 10121), (26, 7736), (24, 7768)],
        "Muslim": [(27, 11207)],
        "No Affiliation": [],
        "Other": [(19, 14223)],
    },
    "ethnicity": {
        "Asian": [
            (19, 2676),
        ],
        "Black": [
            (13, 8247),
            (20, 3022),
            (15, 7906),
            (6, 1949),
            (27, 9625),
            (22, 4804),
            (6, 1949),
        ],
        "Hispanic": [(16, 5887)],
        "Mixed": [(31, 12111)],
        "Other": [(16, 290), (26, 4102), (4, 8338)],
        "White": [
            (18, 1513),
            (7, 13098),
            (14, 1663),
            (7, 5240),
            (12, 9907),
        ],
    },
    "employment_status": {
        "Homemaker / Stay-at-home parent": [
            (15, 5337),
            (15, 4030),
            (16, 10635),
            (7, 11311),
            (14, 3824),
        ],
        "Retired": [(16, 9473), (12, 6809), (14, 3824)],
        "Student": [(20, 4110), (15, 10570), (20, 6742)],
        "Unemployed, not seeking work": [
            (2, 5887),
        ],
        "Unemployed, seeking work": [
            (31, 3252),
        ],
        "Working full-time": [
            (1, 12521),
            (16, 12729),
            (14, 9098),
            (24, 4294),
            (18, 4513),
        ],
        "Working part-time": [
            (22, 7385),
        ],
    },
    "education": {
        "Completed Primary School": [
            (19, 10255),
            (21, 1781),
            (20, 6742),
            (20, 7158),
            (16, 4945),
            (21, 9396),
        ],
        "Completed Secondary School": [
            (16, 1633),
            (18, 1513),
        ],
        "Graduate / Professional degree": [(1, 14096)],
        "Some Primary": [(9, 4594)],
        "Some Secondary": [
            (9, 11428),
            (22, 7971),
            (11, 14254),
            (16, 9087),
            (17, 2782),
        ],
        "Some University but no degree": [
            (13, 9633),
            (14, 3718),
            (14, 9825),
        ],
        "University Bachelors Degree": [
            (4, 8065),
            (25, 9825),
            (16, 14122),
        ],
        "Vocational": [
            (22, 1465),
            (20, 12891),
            (12, 12395),
            (12, 9907),
            (31, 357),
            (14, 1663),
        ],
    },
    "birth_region": {
        "Africa": [
            (21, 10771),
            (18, 10493),
            (13, 8247),
            (28, 13863),
            (15, 7906),
            (6, 1949),
            (27, 9625),
        ],
        "Americas": [
            (20, 6742),
            (14, 3718),
            (19, 10255),
            (9, 5768),
            (10, 4139),
            (16, 8310),
            (18, 10761),
        ],  # many american spelling
        "Asia": [(19, 2676)],
        "Europe": [
            (18, 1513),
            (16, 13518),
            (16, 11662),
            (16, 11662),
            (7, 13098),
            (17, 12173),
        ],
        "Oceania": [
            (7, 3427),
            (29, 9632),
            (6, 12193),
            (31, 11553),
            (16, 290),
            (9, 5716),
        ],
    },
    "reside_region": {
        "Africa": [
            (18, 10493),
            (20, 3022),
        ],  # have not listed all of them, but very similar to birth region
        "Americas": [
            (13, 9633),
            (10, 4139),
            (20, 6742),
            (16, 4945),
            (19, 10255),
        ],
        "Asia": [(19, 2676), (25, 3), (24, 8135)],
        "Europe": [
            (18, 1513),
            (16, 13518),
            (11, 4190),
            (29, 2852),
        ],
        "Oceania": [
            (23, 9359),
            (5, 13524),
            (31, 11553),
            (29, 9632),
        ],
    },
    "marital_status": {
        "Divorced / Separated": [(7, 578)],
        "Married": [(1, 4798), (14, 3824), (19, 6460), (19, 6460)],
        "Never been married": [(0, 3521)],
        "Widowed": [(1, 4798)],
    },
    "english_proficiency": {
        "Advanced": [
            (17, 12173),
            (5, 10888),
            (14, 10950),
            (20, 6742),
        ],
        "Basic": [(2, 4190)],
        "Fluent": [(13, 1626), (9, 5768)],
        "Intermediate": [(2, 4190), (17, 12173), (1, 198)],
        "Native speaker": [
            (17, 10962),
            (9, 533),
            (20, 12891),
            (11, 4190),
            (16, 9473),
            (13, 11171),
        ],
    },
}


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
        default="",
        help="Question dataset to evaluate model on",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        choices=["neurons", "activations", "cross_check_activations", "vocab"],
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "neurons":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        unembed = model.lm_head
        unembed_weight = unembed.weight
        stacked_params = []
        for name, param in model.named_parameters():
            if "mlp.down_proj.weight" in name:  #'mlp.c_proj.weight'
                n_neurons = param.shape[1]
                stacked_params.append(param)
        # elif "mlp.dense_4h_to_h.weight" in name:
        #    stacked_params.append(param.T)
        if not stacked_params:
            raise KeyError(
                "Could not fetch value vectors with predefined keys"
            )
        layers = len(stacked_params)
        unembed_weight = unembed_weight.to(device)

        neurons = {}
        for demographic_col in tqdm(demographic_cols):
            neurons[demographic_col] = {}
            for group, target in tqdm(groups[demographic_col]):
                neurons[demographic_col][group] = []
                sims = []
                for l in range(layers):
                    with open(
                        args.results_dir
                        + f"/{args.model.split('/')[1]}_probe_{demographic_col}_{l}.pkl",
                        "rb",
                    ) as infile:
                        probe = pickle.load(infile)
                    weights = torch.from_numpy(probe.coef_[target]).to(device)
                    params = stacked_params[l].to(device)
                    sims += [
                        F.cosine_similarity(params[:, i], weights, dim=0)
                        for i in range(params.shape[1])
                    ]
                _, top_ind = torch.topk(torch.tensor(sims), 100, largest=True)
                for global_id in top_ind:
                    layer_id = global_id // n_neurons
                    neuron_id = global_id % n_neurons
                    neurons[demographic_col][group].append(
                        (layer_id.item(), neuron_id.item())
                    )
        with open(
            args.results_dir + f"/{args.model.split('/')[1]}_neurons.pkl", "wb"
        ) as outfile:
            pickle.dump(neurons, outfile)

    if args.mode == "activations" or args.mode == "cross_check_activations":
        if args.mode == "activations":
            with open(
                args.results_dir + f"/{args.model.split('/')[1]}_neurons.pkl",
                "rb",
            ) as infile:
                neurons = pickle.load(infile)
            if os.path.isfile(
                args.results_dir
                + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_neuron_activations.pkl"
            ):
                with open(
                    args.results_dir
                    + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_neuron_activations.pkl",
                    "rb",
                ) as infile:
                    neuron_activations = pickle.load(infile)

            else:
                neuron_activations = {}
        else:
            neuron_activations = {}

            neurons = {
                demo: {
                    groups[demo][i][0]: [] for i in range(len(groups[demo]))
                }
                for demo in groups
            }

            for demo in interesting_neurons:
                for group1 in interesting_neurons[demo]:
                    for group2 in neurons[demo]:
                        if group1 != group2:
                            for neuron in interesting_neurons[demo][group1]:
                                neurons[demo][group2].append(neuron)
            print(neurons)

        try:
            special_model = HookedTransformer.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device=device
            )
        except:
            special_model_hf = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
            )
            special_model = HookedTransformer.from_pretrained(
                args.model.split("/")[1],
                hf_model=special_model_hf,
                torch_dtype=torch.bfloat16,
                device=device,
            )
        special_model.tokenizer.padding_side = "left"
        special_model.tokenizer.pad_token_id = (
            special_model.tokenizer.eos_token_id
        )
        if args.dataset:
            df = pd.read_pickle(
                args.data_dir + f"/prism_questions_{args.dataset}.gz",
                compression="gzip",
            )
            df = df[df["age"] != ""]
            questions = [
                "Is it true that arctic sea ice loss is three times greater than antarctic sea ice gain, and the amount of solar energy absorbed by the earth is increasing as a result? Respond with either 'Yes' or 'No' and no additonal text."
            ]
            df = df[df["question"].isin(questions)]
            # with open(args.data_dir + "/questions.pkl", "rb") as infile:
            #     questions = pickle.load(infile)

        else:
            df = pd.read_pickle(
                args.data_dir + "/prism_preprocessed.gz",
                compression="gzip",
            )
        for demographic_col in tqdm(neurons):
            if demographic_col not in neuron_activations:
                neuron_activations[demographic_col] = {}
            for group in tqdm(neurons[demographic_col]):
                if group in neuron_activations[demographic_col]:
                    continue
                neuron_activations[demographic_col][group] = {}
                neurons_group = neurons[demographic_col][group]
                df_group = df[df[demographic_col] == group]
                convos = [
                    (
                        [
                            {
                                "role": turn["role"].replace(
                                    "model", "assistant"
                                ),
                                "content": turn["content"],
                            }
                            for turn in df_group.iloc[i][
                                "conversation_history"
                            ]
                            if turn["role"] == "user"
                            or turn["if_chosen"] == True
                        ]
                        + [
                            {
                                "role": "user",
                                "content": df_group.iloc[i]["question"],
                            }
                        ]
                        if args.dataset
                        else [
                            {
                                "role": turn["role"].replace(
                                    "model", "assistant"
                                ),
                                "content": turn["content"],
                            }
                            for turn in df_group.iloc[i][
                                "conversation_history"
                            ]
                            if turn["role"] == "user"
                            or turn["if_chosen"] == True
                        ]
                    )
                    for i in range(len(df_group))
                ]
                for convo in convos:
                    to_remove = []
                    for i in range(len(convo)):
                        if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                            to_remove.append(i)
                    to_remove = to_remove[::-1]
                    for idx in to_remove:
                        del convo[idx]
                inputs = [
                    special_model.tokenizer.apply_chat_template(
                        convo,
                        tokenize=False,
                        add_generation_prompt=True if args.dataset else False,
                    )
                    for convo in convos
                ]

                prompt_tokens = special_model.to_tokens(inputs).to(device)
                layers_of_interest = [n[0] for n in neurons_group]
                layers_filter = lambda name: name in [
                    utils.get_act_name("mlp_post", l)
                    for l in layers_of_interest
                ]

                post_acts = defaultdict(list)

                for idx in tqdm(
                    range(0, prompt_tokens.shape[0], args.batch_size)
                ):

                    batch = prompt_tokens[idx : idx + args.batch_size]

                    with torch.inference_mode():
                        logits, cache = special_model.run_with_cache(
                            batch, names_filter=layers_filter
                        )
                    for layer, idx in neurons_group:
                        post_act = cache[
                            utils.get_act_name("mlp_post", layer)
                        ][:, -1, idx]
                        post_acts[(layer, idx)].extend(post_act.tolist())

                # Calculate average post activation for individual neuron
                for layer, idx in neurons_group:
                    neuron_activations[demographic_col][group][
                        (layer, idx)
                    ] = np.mean(post_acts[(layer, idx)])

                # post_act_mean_group = np.mean(list(post_act_mean_individual.values()))
                print(neuron_activations)
                with open(
                    args.results_dir
                    + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_neuron_activations{'_cross_check' if args.mode == 'cross_check_activations' else ''}.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(neuron_activations, outfile)

    if args.mode == "vocab":
        if os.path.isfile(
            args.results_dir
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_vocab.pkl"
        ):
            with open(
                args.results_dir
                + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_vocab.pkl",
                "rb",
            ) as infile:
                vocab = pickle.load(infile)

        else:
            vocab = {}
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        unembed = model.lm_head.weight

        with open(
            args.results_dir + f"/{args.model.split('/')[1]}_neurons.pkl", "rb"
        ) as infile:
            neurons = pickle.load(infile)
        with open(
            args.results_dir
            + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_neuron_activations.pkl",
            "rb",
        ) as infile:
            neuron_activations = pickle.load(infile)

        for demographic_col in tqdm(neurons):
            if demographic_col not in vocab:
                vocab[demographic_col] = {}
            for group in tqdm(neurons[demographic_col]):
                if group not in vocab[demographic_col]:
                    vocab[demographic_col][group] = {}
                filtered_neurons = [
                    n
                    for n in neurons[demographic_col][group]
                    if neuron_activations[demographic_col][group][n] > 0
                ]
                for layer_id, neuron_id in filtered_neurons:
                    if (layer_id, neuron_id) in vocab[demographic_col][group]:
                        continue
                    down_column = model.model.layers[
                        layer_id
                    ].mlp.down_proj.weight.T[neuron_id]
                    assert unembed.size(1) == down_column.size(0)

                    projection = unembed @ down_column
                    _, sorted_indices = torch.sort(projection, descending=True)

                    interp = []
                    for ind in sorted_indices[:20]:
                        interp.append(tokenizer.decode(ind))
                    vocab[demographic_col][group][
                        (layer_id, neuron_id)
                    ] = interp
                print(vocab)
                with open(
                    args.results_dir
                    + f"/{args.model.split('/')[1]}{'_'+args.dataset if args.dataset else ''}_vocab.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(vocab, outfile)
