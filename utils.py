import os
from lib2to3.pgen2.tokenize import tokenize

import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer


def GetReports(labels, text_encoder, tokenizer):
    map_organ = {'BRAIN': 1, 'HEAD': 1, 'KNEE': 2, 'CSPINE': 3, 'LSPINE': 4, 'SPINE': 5,
                'TSPINE': 6, 'CAROTID': 7, 'SHOULDER': 8, 'UTERUS': 9, 'lp': 10,
                'EMPTY': 0}
    # print(labels.shape)

    prompt_embeds = torch.empty((0,))
    for i in range(0, labels.shape[0]):
        reports = "MRI image acquired by facilities from "
        if labels[i][0] == 1:  # GE
            reports += "GE. "
        elif labels[i][0] == 2:
            reports += "Philip. "
        elif labels[i][0] == 3:
            reports += "Siemens. "
        else:
            reports += "Siemens. "
        reports += "Organ in image is "
        for organ_name, value in map_organ.items():
            if labels[i][1] == value:
                reports += organ_name
                break
        reports += "."
        reports_tokens = tokenizer(reports, return_tensors="pt")
        outputs = text_encoder(**reports_tokens)
        prompt_embeds_cur = outputs.last_hidden_state
        # print(reports)
        # print(prompt_embeds_cur.shape)
        prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_cur), dim=0)

    return prompt_embeds
