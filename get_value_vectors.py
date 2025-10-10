import argparse
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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
        "--folder",
        type=str,
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation/"
    )
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unembed = model.lm_head
    unembed_weight = unembed.weight
    stacked_params = []
    for name, param in model.named_parameters():
        if "mlp.c_proj.weight" in name:
            stacked_params.append(param)
        elif "mlp.dense_4h_to_h.weight" in name:
            stacked_params.append(param.T)
    if not stacked_params:
        raise KeyError("Could not fetch value vectors with predefined keys")
    layers = model.config.n_layer
    big_matrix = torch.cat(stacked_params, dim=0).to(device)
    n_neurons = big_matrix.shape[0] // layers
    big_matrix = torch.split(big_matrix, [n_neurons] * layers)
    unembed_weight = unembed_weight.to(device)

    imp_columns = {}
    for demographic_col in demographic_cols:
        imp_columns[demographic_col] = {}
        for group, target in groups[demographic_col]:
            imp_columns[demographic_col][group] = []
            sims = []
            for l in layers:
                with open(
                    args.folder
                    + f"{args.model.split('/')[1]}_probe_{demographic_col}_{l}.pkl",
                    "rb",
                ) as infile:
                    probe = pickle.load(infile)
                weights = probe.coef_[target].to(device)
                sims.append(F.cosine_similarity(big_matrix[l], weights))
            _, top_ind = torch.topk(sims.unsqueeze(), 100, largest=True)
            for global_id in top_ind:
                layer_id = global_id // n_neurons
                neuron_id = global_idd % n_neurons
                imp_columns[demographic_col][group].append(
                    (layer_id.item(), neuron_id.item())
                )
    print(imp_columns)
