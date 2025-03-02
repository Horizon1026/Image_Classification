import torch
from transformer import Transformer

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
        self.transformer = Transformer(
            dim_embed_token = dim,
            dim_hidden_layer = mlp_dim,
            num_heads = heads,
            num_layers = depth,
            dropout = dropout)
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


if __name__ == '__main__':
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=6,
        heads=8,
        mlp_dim=768 * 4,
        dropout=0.1,
        emb_dropout=0.1
    )
    img = torch.randn(1, 3, 224, 224)
    preds = model(img)
    print(preds.size())
