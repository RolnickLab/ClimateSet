
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from emulator.src.utils.pos_embed import (get_1d_sincos_pos_embed_from_grid,
                                 get_2d_sincos_pos_embed)
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_


class TokenizedBase(nn.Module):
    """Base model for tokenized MAE and tokenized ViT Including patch embedding and encoder."""

    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        init_mode="xavier",  # xavier or small
        in_vars=["pr", "tas"],
        channel_agg="mean"
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.init_mode = init_mode
        self.in_vars = in_vars

        # separate linear layers to embed each token, which is 1xpxp
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(in_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # positional embedding and channel embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb)
        self.channel_embed, self.channel_map = self.create_channel_embedding(learn_pos_emb, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if channel_agg == "mean":
            self.channel_agg = None
        elif channel_agg == "attention":
            self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        else:
            raise NotImplementedError

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm
                ) #drop=drop rate #DEPRECATED - clarify if we wanna use proj_drop or attn_drop
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

    def create_channel_embedding(self, learnable, dim):
        channel_embed = nn.Parameter(torch.zeros(1, len(self.in_vars), dim), requires_grad=learnable)
        # TODO: create a mapping from var --> idx
        channel_map = {}
        idx = 0
        for var in self.in_vars:
            channel_map[var] = idx
            idx += 1
        return channel_embed, channel_map

    def get_channel_ids(self, vars):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids)

    def get_channel_emb(self, channel_emb, vars):
        ids = self.get_channel_ids(vars)
        return channel_emb[:, ids, :]

    def initialize_weights(self ):
        # initialization
        # initialize pos_emb and channel_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.in_vars))
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            else:
                trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        pass

    def forward_loss(self, x):
        pass

    def forward(self, x):
        pass


# model = TokenizedMAE(depth=4, decoder_depth=2).cuda()
# x = torch.randn(2, 3, 128, 256).cuda()
# loss, pred, mask = model(x)
# print (loss)