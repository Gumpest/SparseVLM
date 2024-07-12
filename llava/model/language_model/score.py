import torch
import torch.nn as nn
import torch.nn.functional as F


def attn_vis_text_infer(vis, text):
    '''
    vis: B, L_1, C
    text: B, L_2, C
    '''
    # Compute attention weights
    attn_weights = torch.matmul(vis, text.transpose(1, 2))  # B, L_1, L_2

    # Normalize attention in each text token
    attn_weights = torch.softmax(attn_weights, dim=1)  # B, L_1, L_2

    attn_weights = attn_weights.mean(2) # B, L1

    thresh = attn_weights.mean()
    # thresh = attn_weights.median()
    # print(attn_weights, thresh)

    return torch.where(attn_weights >= thresh, 1, 0)


def attn_vis_text(vis, text, prev_decision):
    '''
    vis: B, L_1, C
    text: B, L_2, C
    prev_decision: B, L_1
    '''
    # Compute attention weights
    attn_weights = torch.matmul(vis, text.transpose(1, 2))  # B, L_1, L_2

    # Normalize attention in each text token
    bias = torch.zeros(vis.shape[0], vis.shape[1], dtype=attn_weights.dtype).to(device=attn_weights.device)
    bias = bias.masked_fill_(prev_decision.logical_not(), float("-inf")).unsqueeze(2)

    attn_weights = attn_weights + bias

    attn_weights = torch.softmax(attn_weights, dim=1)  # B, L_1, L_2

    attn_weights = attn_weights.mean(2)

    thresh = attn_weights.mean()

    return torch.where(attn_weights >= thresh, 1, 0)


def attn_postprocess(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]
    # self_attn_weights = self_attn_weights.max(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1
    relation_vis_text = relation_vis_text.mean(1) # B, L1

    thresh = relation_vis_text.mean(1) # B
    if layer_idx == 1:
        thresh = 0.6 * thresh

    mask = []
    for b in range(self_attn_weights.shape[0]):
        mask.append(torch.where(relation_vis_text[b] >= thresh[b], 1, 0))
    mask = torch.stack(mask, dim=0)

    return mask


if __name__ == "__main__":

    # torch.manual_seed(1)
    # a = torch.rand(1, 6, 1024)
    # b = torch.concat([torch.rand(1, 60, 1024), torch.zeros(1, 35, 1024)], dim=1)
    # mask = torch.Tensor([[1, 1, 1, 1, 1, 0]])
    # res = attn_vis_text(a, b, mask)
    # print(res)

    self_attn_weights, v_token_start, v_token_num, text_token_start = torch.rand(4, 16, 1084, 1084), 36, 576, 700
    mask = attn_postprocess(self_attn_weights, v_token_start, v_token_num, text_token_start)
    print(mask.shape)