from baukit import TraceDict
from utils import probe_targets
import torch
from torch import nn
from train_probe import get_tokenized_chat
from tqdm import tqdm


def get_layer_names(model):
    which_layers = []  # Which layer/s to intervene
    from_idx = 14
    to_idx = 15
    for name, module in model.named_modules():
        if name != "" and name[-1].isdigit():
            layer_num = name[
                name.rfind("model.layers.") + len("model.layers.") :
            ]
            if from_idx <= int(layer_num) < to_idx:
                which_layers.append(name)
    return which_layers


def optimize_one_inter_rep(
    inter_rep, layer_name, target, probe, mult, normalized=False
):
    global first_time
    tensor = (inter_rep.clone()).to("cuda").requires_grad_(True)
    rep_f = lambda: tensor
    probe_weights = torch.from_numpy(probe.coef_[target]).to("cuda")

    # cur_input_tensor = rep_f().clone().detach()

    if normalized:
        cur_input_tensor = (
                rep_f() - probe_weights * mult * 100 / rep_f().norm()

    else:
        cur_input_tensor = rep_f() - probe_weights * mult

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
            cf_target,
            probe,
            mult=mult,
            normalized=False,
        )
    output[0][:, -1] = cloned_inter_rep.to(torch.float16)
    return output


def modified_model(
    model,
    probes,
    modified_layer_names,
    demographic,
    target,
    batch_size,
    question_convos,
    N,
):
    global probes_dict
    probes_dict = probes
    global mult
    mult = N
    global cf_target
    cf_target = target
    with TraceDict(
        model.model,
        modified_layer_names,
        edit_output=edit_inter_rep_multi_layers,
    ) as ret:
        model_answer = [
            answer[0]["generated_text"][-1]["content"]
            for answer in tqdm(
                model(
                    question_convos,
                    batch_size=batch_size,
                    do_sample=False,
                    max_new_tokens=100,
                ),
                total=len(question_convos),
            )
        ]
    return model_answer
