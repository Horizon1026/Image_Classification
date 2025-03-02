import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_embed_token, dim_single_qk, dim_single_v, dim_out, num_heads, dropout = 0):
        super().__init__()
        # Validate parameters.
        assert num_heads > 0, 'Num of head must be larger than 0.'
        assert num_heads > 1 or (num_heads == 1 and dim_single_v == dim_out), 'Since num of head is 1, scaled dot-product self attention will be actived, whose dim_out should be equal to dim_single_v.'
        # Initialize parameters and layers.
        self.num_heads = num_heads
        self.dim_single_qk = dim_single_qk
        self.dim_single_v = dim_single_v
        self.dim_single_head = dim_single_qk * 2 + dim_single_v
        self.qk_scale = dim_single_qk ** -0.5
        self.softmax = torch.nn.Softmax(dim = -1) # Do normalization of each col.
        # Use only one MLP to obtain q, k, v of all heads.
        self.to_qkv = torch.nn.Linear(dim_embed_token, self.dim_single_head * num_heads, bias = False)
        # If num_heads == 1, this will be scaled dot-product self attention.
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(dim_single_v * num_heads, dim_out),
            torch.nn.Dropout(dropout),
        ) if num_heads != 1 else torch.nn.Identity()

    def forward(self, x):
        # x = [ batch_size, num_embed_tokens, dim_embed_token ]
        # all_qkv = [ q1 k1 v1 | q2 k2 v2 | ... | qn kn vn ]
        all_qkv = self.to_qkv(x)
        # self_attentions = [ attention 1 | attention 2 | ... | attention n ]
        self_attentions = []
        for i in range(self.num_heads):
            offset = i * self.dim_single_head
            q = all_qkv[:, :, offset : offset + self.dim_single_qk]
            offset = offset + self.dim_single_qk
            k = all_qkv[:, :, offset : offset + self.dim_single_qk]
            offset = offset + self.dim_single_qk
            v = all_qkv[:, :, offset : offset + self.dim_single_v]
            # score_matrix = softmax(q * k.transpose())
            score_matrix = torch.matmul(q, k.transpose(-1, -2)) * self.qk_scale
            score_matrix = self.softmax(score_matrix)
            self_attention = torch.matmul(score_matrix, v)
            self_attentions.append(self_attention)
        # Concate all self attentions.
        concated_attention = torch.cat(self_attentions, dim = -1)
        return self.to_out(concated_attention)

class SelfDotProductAttention(torch.nn.Module):
    def __init__(self, dim_embed_token, dim_single_qk, dim_single_v, dropout = 0):
        super().__init__()
        self.attention = MultiHeadAttention(dim_embed_token, dim_single_qk, dim_single_v, dim_single_v, num_heads = 1, dropout = dropout)
    def forward(self, x):
        return self.attention(x)


if __name__ == '__main__':
    print('>> Test attention.')
    batch_size = 5
    num_embed_token = 100
    dim_embed_token = 10
    dim_single_qk = 12
    dim_single_v = 15
    dim_out = 20
    num_heads = 5
    dropout = 0

    print('>> Test multi-head attention.')
    model = MultiHeadAttention(
        dim_embed_token,
        dim_single_qk,
        dim_single_v,
        dim_out,
        num_heads,
        dropout
    )
    input = torch.randn(batch_size, num_embed_token, dim_embed_token)
    print(input.size())
    output = model(input)
    print(output.size())

    print('>> Test self dot-product attention.')
    model = SelfDotProductAttention(
        dim_embed_token,
        dim_single_qk,
        dim_single_v,
        dropout
    )
    input = torch.randn(batch_size, num_embed_token, dim_embed_token)
    print(input.size())
    output = model(input)
    print(output.size())
