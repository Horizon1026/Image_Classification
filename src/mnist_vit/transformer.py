import torch
from mnist_vit.attention import *

class FeedForward(torch.nn.Module):
    def __init__(self, dim_token, dim_hidden_layer, dim_out, dropout = 0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_token, dim_hidden_layer),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_hidden_layer, dim_out),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim_token, dim_hidden_layer, num_heads, dropout = 0):
        super().__init__()
        self.norm_attention = torch.nn.LayerNorm(dim_token)
        self.attention = SelfDotProductAttention(
            dim_token = dim_token,
            dim_single_qk = dim_token,
            dim_single_v = dim_token,
            dropout = dropout
        ) if num_heads == 1 else MultiHeadAttention(
            dim_token = dim_token,
            dim_single_qk = dim_token,
            dim_single_v = dim_token,
            dim_out = dim_token,
            num_heads = num_heads,
            dropout = dropout
        )
        self.norm_feed_forward = torch.nn.LayerNorm(dim_token)
        self.feed_forward = FeedForward(
            dim_token = dim_token,
            dim_hidden_layer = dim_hidden_layer,
            dim_out = dim_token,
            dropout = dropout
        )
    def forward(self, x):
        x = self.norm_attention(x)
        x = self.attention(x) + x
        x = self.norm_feed_forward(x)
        x = self.feed_forward(x) + x
        return x


if __name__ == '__main__':
    print('>> Test transformer.')
    batch_size = 5
    num_embed_token = 100
    dim_token = 10
    dim_single_qk = 12
    dim_single_v = 15
    dim_out = 20
    num_heads = 5
    dropout = 0

    print('>> Test transformer encoder with single layer.')
    model = TransformerEncoder(
        dim_token = dim_token,
        dim_hidden_layer = 32,
        num_heads = num_heads,
        dropout = dropout
    )
    input = torch.randn(batch_size, num_embed_token, dim_token)
    print(input.size())
    output = model(input)
    print(output.size())
