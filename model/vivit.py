import torch
import torch.nn as nn
import einops
import torch.nn.functional as F


class ViViT(nn.Module):
    def __init__(self, num_classes,
                 size=(224, 224), frame_per_clip=32,
                 t=2, h=16, w=16, n_head=12, n_layer=12, patch_dim=512, d_feature=2048):
        super(ViViT, self).__init__()

        self.t = t
        self.h = h
        self.w = w
        self.n_t = frame_per_clip // t
        self.n_h = size[0] // h
        self.n_w = size[1] // w
        self.N = self.n_t * self.n_h * self.n_w

        self.embed_projection = nn.Linear(self.t * self.h * self.w * 3, patch_dim)
        self.encoder_layer = [FactorisedTransformerLayer(self.n_t, self.n_h, self.n_w, n_head,
                                                         d_model=patch_dim,
                                                         d_feature=d_feature)
                              for _ in range(n_layer)]
        self.encoder_layer = nn.Sequential(*self.encoder_layer)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, x):
        # x: (B,C,N,H,W)
        x = einops.rearrange(x, "b c (n_t t) (n_h h) (n_w w) -> b n_t n_h n_w (t h w c)",
                             t=self.t, h=self.h, w=self.w)
        token = self.embed_projection(x)
        token = self.encoder_layer(token)
        # mlp head
        token = einops.rearrange(token, "n n_t n_h n_w d_model->n (n_t n_h n_w) d_model")
        token = torch.mean(token, dim=1)
        logits = self.mlp_head(token)
        return logits


class FactorisedTransformerLayer(nn.Module):
    def __init__(self, n_t=2, n_h=16, n_w=16, n_head=12, d_model=3072, d_feature=2048):
        super(FactorisedTransformerLayer, self).__init__()
        self.attention = FactorisedDotProductAttention(n_head, d_model,
                                                       d_k=d_model // n_head,
                                                       d_v=d_model // n_head)
        self.n_t = n_t
        self.n_h = n_h
        self.n_w = n_w
        self.projection = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffc1 = nn.Linear(d_model, d_feature)
        self.ffc2 = nn.Linear(d_feature, d_model)
        self.relu = nn.ReLU()

    def forward(self, token):
        # factorised dot-product attention
        token_residual = token
        token = self.layer_norm1(token)
        n_t, n_h, n_w = token.size(1), token.size(2), token.size(3)
        temporal = einops.rearrange(token, "b n_t n_h n_w d -> (b n_t) (n_h n_w) d")
        spatial = einops.rearrange(token, "b n_t n_h n_w d -> (b n_h n_w) n_t d")
        temporal, spatial = self.attention(temporal, spatial)
        temporal = einops.rearrange(temporal, "(b n_t) (n_h n_w) d -> b n_t n_h n_w d",
                                    n_t=n_t, n_h=n_h, n_w=n_w)
        spatial = einops.rearrange(spatial, "(b n_h n_w) n_t d -> b n_t n_h n_w d",
                                   n_t=n_t, n_h=n_h, n_w=n_w)
        y = torch.cat([temporal, spatial], dim=-1)
        y = self.projection(y)
        y = y + token_residual
        # mlp
        y_residual = y
        y = self.layer_norm2(y)
        y = self.ffc2(self.relu(self.ffc1(y))) + y_residual
        return y


class FactorisedDotProductAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(FactorisedDotProductAttention, self).__init__()

        assert n_head % 2 == 0

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.w_qt = nn.Linear(d_model, (n_head // 2) * d_k)
        self.w_kt = nn.Linear(d_model, (n_head // 2) * d_k)
        self.w_vt = nn.Linear(d_model, (n_head // 2) * d_v)
        self.w_qs = nn.Linear(d_model, (n_head // 2) * d_k)
        self.w_ks = nn.Linear(d_model, (n_head // 2) * d_k)
        self.w_vs = nn.Linear(d_model, (n_head // 2) * d_v)
        self.temporal_attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.spatial_attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def calculate_attention(self, x, dim):
        bs = x.size(0)
        n = x.size(1)

        if dim == "temporal":
            q = self.w_qt(x).view(bs, n, self.n_head // 2, self.d_k).transpose(1, 2)
            k = self.w_kt(x).view(bs, n, self.n_head // 2, self.d_k).transpose(1, 2)
            v = self.w_vt(x).view(bs, n, self.n_head // 2, self.d_v).transpose(1, 2)
            x, attention = self.temporal_attention(q, k, v)
        elif dim == "spatial":
            q = self.w_qs(x).view(bs, n, self.n_head // 2, self.d_k).transpose(1, 2)
            k = self.w_ks(x).view(bs, n, self.n_head // 2, self.d_k).transpose(1, 2)
            v = self.w_vs(x).view(bs, n, self.n_head // 2, self.d_v).transpose(1, 2)
            x, attention = self.spatial_attention(q, k, v)
        else:
            raise ValueError
        x = x.transpose(1, 2).contiguous().view(bs, n, -1)
        return x, attention

    def forward(self, temporal, spatial):
        temporal, attention_t = self.calculate_attention(temporal, dim="temporal")
        spatial, attention_s = self.calculate_attention(spatial, dim="spatial")
        return temporal, spatial


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
