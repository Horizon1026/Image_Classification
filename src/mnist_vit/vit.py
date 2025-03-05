import torch
from transformer import TransformerEncoder

class VitModule(torch.nn.Module):
    def __init__(self, image_size, patch_size, dim_token,
                 dim_hidden_layer, num_heads, num_layers, num_classes, dropout = 0):
        super().__init__()
        assert len(image_size) == 3, 'Image size must be list of [channels, rows, cols]'
        assert len(patch_size) == 2, 'Patch size must be list of [rows, cols]'
        image_channels, image_rows, image_cols = image_size
        patch_rows, patch_cols = patch_size
        num_tokens = (image_rows // patch_rows) * (image_cols // patch_cols)

        self.patch_embedding = torch.nn.Conv2d(
            in_channels = image_channels,
            out_channels = dim_token,
            kernel_size = [patch_rows, patch_cols],
            stride = [patch_rows, patch_cols],
            padding = 0,
        )
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_tokens, dim_token))
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.transformer_encoders = torch.nn.ModuleList([
            TransformerEncoder(
                dim_token = dim_token,
                dim_hidden_layer = dim_hidden_layer,
                num_heads = num_heads,
                dropout = dropout
            )
            for _ in range(num_layers)
        ])
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_token),
            torch.nn.Linear(dim_token, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Embedding layer.
        # x = [batch_size, image_channels, image_rows, image_cols]
        x = self.patch_embedding(x)
        # x = [batch_size, dim_token, sqrt(num_tokens), sqrt(num_tokens)]
        x = x.flatten(2).transpose(-1, -2)
        # x = [batch_size, num_tokens, dim_token]
        for i in range(batch_size):
            # x = (x)[batch_size, num_tokens, dim_token] + (pos)[1, num_tokens, dim_token]
            x[i] = x[i] + self.pos_embedding
        x = self.embedding_dropout(x)

        # Transformer layer.
        for layer in self.transformer_encoders:
            x = layer(x)
        # x = [batch_size, num_tokens, dim_token]
        x = x.mean(dim = 1)
        # x = [batch_size, dim_token]

        # MLP header.
        x = self.mlp_head(x)
        # x = [batch_size, num_classes]
        return x


if __name__ == '__main__':
    batch_size = 100
    image_size = [1, 28, 28]
    patch_size = [8, 8]

    model = VitModule(
        image_size = image_size,
        patch_size = patch_size,
        dim_token = image_size[0] * patch_size[0] * patch_size[1],
        dim_hidden_layer = 256,
        num_heads = 3,
        num_layers = 5,
        num_classes = 10,
        dropout = 0,
    )
    input = torch.randn(batch_size, image_size[0], image_size[1], image_size[2])
    print(input.size())
    output = model(input)
    print(output.size())
