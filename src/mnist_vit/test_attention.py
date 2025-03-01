import torch
from attention import MultiHeadAttention

if __name__ == '__main__':
    print('>> Test attention.')
    batch_size = 5
    num_embed_token = 100
    dim_embed_token = 10
    dim_single_qk=15
    dim_single_v=18
    num_heads=5
    dim_out=20
    dropout=0

    model = MultiHeadAttention(
        dim_embed_token,
        dim_single_qk,
        dim_single_v,
        num_heads,
        dim_out,
        dropout
    )
    input = torch.randn(batch_size, num_embed_token, dim_embed_token)
    print(input.size())
    output = model(input)
    print(output.size())
