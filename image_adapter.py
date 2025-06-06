import torch
from torch import nn


# File: image_adapter.py
# Author: fancyfeast
# Modified by: nflamously
# Original License: Apache License 2.0 / unknown
# Changes:
# * optimize to use float16 instead
# This code was originally authored by fancyfeast. All modifications are documented and follow the terms of the original license.

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int,
                 deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features, dtype=torch.float16)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features, dtype=torch.float16)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features, dtype=torch.float16)
        self.pos_emb = None if not pos_emb else nn.Parameter(
            torch.zeros(num_image_tokens, input_features, dtype=torch.float16))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features, dtype=torch.float16)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[
                -1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (
            x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)
