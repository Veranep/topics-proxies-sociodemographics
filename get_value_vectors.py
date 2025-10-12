import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM
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
        "--data_dir",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation/"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation/"
    )
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        choices=["neurons", "activations", "vocab"],
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

    if args.mode == "activations":
        with open(
            args.results_dir + f"/{args.model.split('/')[1]}_neurons.pkl", "rb"
        ) as infile:
            neurons = pickle.load(infile)

        neuron_activations = {}

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
        df = pd.read_pickle(
            args.data_dir + "prism_preprocessed.gz",
            compression="gzip",
        )
        for demographic_col in tqdm(neurons):
            neuron_activations[demographic_col] = {}
            for group in tqdm(neurons[demographic_col]):
                neuron_activations[demographic_col][group] = {}
                neurons_group = neurons[demographic_col][group]
                df_group = df[df[demographic_col] == group]
                convos = [
                    [
                        {
                            "role": turn["role"].replace("model", "assistant"),
                            "content": turn["content"],
                        }
                        for turn in convo
                        if turn["role"] == "user" or turn["if_chosen"] == True
                    ]
                    for convo in df_group["conversation_history"].tolist()
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
                        add_generation_prompt=True,
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
            + f"/{args.model.split('/')[1]}_neuron_activations.pkl",
            "wb",
        ) as outfile:
            pickle.dump(neuron_activations, outfile)

    if args.mode == "vocab":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        unembed = model.lm_head.weight

        vocab = {}

        with open(
            args.results_dir + f"/{args.model.split('/')[1]}_neurons.pkl", "rb"
        ) as infile:
            neurons = pickle.load(infile)
        with open(
            args.results_dir
            + f"/{args.model.split('/')[1]}_neuron_activations.pkl",
            "rb",
        ) as infile:
            neuron_activations = pickle.load(infile)

        for demographic_col in tqdm(neurons):
            vocab[demographic_col] = {}
            for group in tqdm(neurons[demographic_col]):
                vocab[demographic_col][group] = {}
                filtered_neurons = [
                    n
                    for n in neurons[demographic_col][group]
                    if neuron_activations[demographic_col][group] > 0
                ]
                for layer_id, neuron_id in filtered_neurons:
                    down_column = model.transformer.h[
                        layer_id
                    ].mlp.down_proj.weight[neuron_id]
                    assert unembed.size(1) == down_column.size(0)

                    projection = unembed @ down_column
                    _, sorted_indices = torch.sort(projection, descending=True)

                    interp = []
                    for ind in sorted_indices[:20]:
                        interp.append(tokenizer.decode(ind))
                    vocab[demographic_col][group][
                        (layer_id, neuron_id)
                    ] = interp
        with open(
            args.results_dir + f"/{args.model.split('/')[1]}_vocab.pkl",
            "wb",
        ) as outfile:
            pickle.dump(vocab, outfile)
