import torch
from attention import MultiHeadAttention

class MlpHeader(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads, dropout = 0):
        super().__init__()
        # inner_dim = dim_head * heads
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = torch.nn.Softmax(dim = -1)
        self.to_qkv = torch.nn.Linear(dim, dim * 3, bias = False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Dropout(dropout),
        ) if project_out else torch.nn.Identity()
    def forward(self, x):
        B, N, C = x.shape
        # qkv : -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape : -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute : -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # Cannot use tensor as tuple, because it is not friendly for torch script.
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout)),
                PreNorm(dim, MlpHeader(dim, mlp_dim, dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

def Pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(torch.nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool = 'cls', channels = 3, dropout = 0, emb_dropout = 0):
        super().__init__()
        image_height, image_width = Pair(image_size)
        patch_height, patch_width = Pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'Pool type must be either cls token or mean pooling.'

        self.to_patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height),
        )
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = torch.nn.Identity()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img).flatten(2).transpose(1, 2)
        _, n, _ = x.shape
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x
