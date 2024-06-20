import torch
import torch.nn as nn

import math

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        # x = self.out_conv(x)
        
        # return x
        return self.out_conv(x)
    
def  batch_index_select(x, idx):

    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def softmax_with_policy_for_training(self, query_states, key_states, value_states, attention_mask, query_length, policy, eps=1e-6):    # attn : [2, 687, 32, 32] policy : [2, 687, 1]
    if attention_mask is not None:
        B, N = attention_mask.size()
        attn_bias = torch.zeros(N, N, dtype=query_states.dtype)
        temp_mask = torch.ones(N, N, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query_states.dtype)
    attn = (query_states @ key_states.transpose(-2, -1))        # [2, 32, 687, 687]
    attn += attn_bias.to(device=query_states.device)
    B, N, _ = policy.size()
    B, H, N, N = attn.size()
    attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
    eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N) # [1, 1, 687, 687]
    attn_policy = attn_policy + (1.0 - attn_policy) * eye   # [2, 1, 687, 687]
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]  # [2, 32, 687, 1]
    attn = attn - max_att   # 
    # attn = attn.exp_() * attn_policy
    # return attn / attn.sum(dim=-1, keepdim=True)

    # for stable training
    attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)  # [2, 32, 687, 687]
    attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)      # [2, 32, 687, 687]
    return attn.type_as(max_att)


def softmax_with_policy(attn, policy, eps=1e-6):    # attn : [2, 687, 32, 32] policy : [2, 687, 1]
    B, N, _ = policy.size()
    B, H, T, N = attn.size()
    if T == 1:
        # attn_policy = policy.reshape(B, 1, 1, N)
        # max_att = torch.max(attn, dim=-1, keepdim=True)[0]  # [2, 32, 687, 1]
        # attn = attn - max_att
        # attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)  # [2, 32, 687, 687]
        # attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)      # [2, 32, 687, 687]
        policy_bias = torch.zeros(B, 1, N, 1, dtype=policy.dtype).to(device=policy.device)
        policy_bias.masked_fill_(policy.logical_not(), float("-inf"))
        policy_bias = policy_bias.permute(0, 1, 3, 2).to(policy.dtype)
        attn += policy_bias.to(device=attn.device)
        attn = torch.softmax(attn, dim=-1)
        return attn
    else:
        # attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
        # eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N) # [1, 1, 687, 687]
        # attn_policy = attn_policy + (1.0 - attn_policy) * eye   # [2, 1, 687, 687]
        # max_att = torch.max(attn, dim=-1, keepdim=True)[0]  # [2, 32, 687, 1]
        # attn = attn - max_att   # 
        # # attn = attn.exp_() * attn_policy
        # # return attn / attn.sum(dim=-1, keepdim=True)

        # # for stable training
        # attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)  # [2, 32, 687, 687]
        # attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)      # [2, 32, 687, 687]
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N) # [1, 1, 687, 687]
        attn_policy = attn_policy + (1.0 - attn_policy) * eye   # [2, 1, 687, 687]
        policy_bias = torch.zeros(B, 1, N, N, dtype=attn_policy.dtype).to(device=attn_policy.device)
        policy_bias.masked_fill_(attn_policy.logical_not(), float("-inf"))
        policy_bias.to(attn_policy.dtype)
        attn += policy_bias
        attn = torch.softmax(attn, dim=-1)
        return attn


def scaled_dot_product_attention_with_policy(query, key, value, policy, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None): 
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if is_causal:
        assert attn_mask is None
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.shape, dtype=query.dtype).to(device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(device=query.device)
    attn_weight = softmax_with_policy(attn_weight, policy)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value