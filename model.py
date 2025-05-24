import torch
import torch.nn as nn

class Vit(nn.Module):
    def __init__(self, img_size=128, patch_size=8, depth=6, num_classes=4,
                 in_channels=1, dim=128, mlp=512, nheads=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nheads,
                                                   dim_feedforward=mlp, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        cls_out = x[:, 0]
        avg_out = x[:, 1:].mean(dim=1)
        x = self.norm(cls_out + avg_out)
        return self.mlp_head(x)
