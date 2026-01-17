from baukit import TraceDict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def get_layer_names(model, from_idx, to_idx):
    which_layers = []  # Which layer/s to intervene
    for name, module in model.named_modules():
        if name != "" and name[-1].isdigit():
            layer_num = name[
                name.rfind("model.layers.") + len("model.layers.") :
            ]
            if from_idx <= int(layer_num) < to_idx:
                which_layers.append(name)
    return which_layers


def optimize_one_inter_rep(
    inter_rep,
    layer_name,
    probe,
):
    global first_time
    tensor = (
        (inter_rep.clone()).to(torch.float64).to("cuda").requires_grad_(True)
    )
    rep_f = lambda: tensor
    probe_weights = (
        torch.from_numpy(probe.coef_[0]).to(torch.float64).to("cuda")
    )
    probe_intercept = torch.from_numpy(np.array(probe.intercept_[0])).to(
        "cuda"
    )

    # cur_input_tensor = rep_f().clone().detach()

    # + made differences a lot worse, - also made them slightly worse

    # next try: rep_f() + (probe_weights - rep_f())

    print(rep_f().shape, probe_weights.shape)

    logits = (
        torch.tensor(
            [
                torch.dot(rep_f().squeeze()[i], probe_weights)
                + probe_intercept
                for i in range(len(rep_f()))
            ]
        )
        .unsqueeze(1)
        .to("cuda")
    )

    W_norm_sq = torch.dot(probe_weights, probe_weights)

    cur_input_tensor = (
        rep_f() - (logits / W_norm_sq) * probe_weights
    ).unsqueeze(0)

    # cur_input_tensor = rep_f() + (probe_weights - rep_f()) * mult

    return cur_input_tensor.clone()


def edit_inter_rep_multi_layers(output, layer_name):
    layer_num = int(
        layer_name[layer_name.rfind("model.layers.") + len("model.layers.") :]
    )
    probe = probes_dict[layer_num]
    cloned_inter_rep = (
        output[0][:, -1].unsqueeze(0).detach().clone().to(torch.float)
    )
    with torch.enable_grad():
        cloned_inter_rep = optimize_one_inter_rep(
            cloned_inter_rep,
            layer_name,
            probe,
        )
    output[0][:, -1] = cloned_inter_rep.to(torch.float16)
    return output


def modified_model(
    model,
    probes,
    modified_layer_names,
    demographic,
    batch_size,
    tokens,
    question_convos,
):
    global probes_dict
    probes_dict = probes
    with TraceDict(
        model.model,
        modified_layer_names,
        edit_output=edit_inter_rep_multi_layers,
    ) as ret:
        model_answer = [
            answer[0]["generated_text"].lower()
            for answer in tqdm(
                model(
                    question_convos,
                    batch_size=batch_size,
                    do_sample=False,
                    max_new_tokens=tokens,
                    return_full_text=False,
                ),
                total=len(question_convos),
            )
        ]
    return model_answer
