import torch
from mnist_vit.transformer import *

class ViTNet(torch.nn.Module):
    def __init__(self, image_size, patch_size, dim_token,
                 dim_hidden_layer, num_heads, num_layers, num_classes, dropout = 0, use_class_token = False):
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
        self.use_class_token = use_class_token
        self.pos_embedding = torch.nn.Parameter(torch.randn(1,
            num_tokens + 1 if use_class_token else num_tokens,
            dim_token))
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
        # x: [batch_size, 1, 28, 28]
        batch_size = x.size(0)

        # 1. Embedding layer. (For example, patch_size = [8, 8])
        # self.patch_embedding: Conv2d(1, dim_token, 8, 8) -> [batch_size, dim_token, 3, 3]
        x = self.patch_embedding(x)

        # Flatten and Transpose: [batch_size, dim_token, 9] -> [batch_size, 9, dim_token]
        # (num_tokens = 3 * 3 = 9)
        x = x.flatten(2).transpose(-1, -2)

        if self.use_class_token:
            # class_token: [batch_size, 1, dim_token]
            class_token = torch.zeros(batch_size, 1, x[0].size(-1), device = x.device)
            # x: [batch_size, 10, dim_token]
            x = torch.cat([class_token, x], dim = 1)

        # Add Position Embedding:
        # x: [batch_size, 9, dim_token] (or [batch_size, 10, dim_token] if use_class_token)
        for i in range(batch_size):
            x[i] = x[i] + self.pos_embedding
        x = self.embedding_dropout(x)

        # 2. Transformer layer.
        for layer in self.transformer_encoders:
            # x: [batch_size, num_tokens(+1), dim_token]
            x = layer(x)

        # Pooling:
        # If use_class_token: pick first token -> [batch_size, dim_token]
        # Else: mean pooling over tokens -> [batch_size, dim_token]
        x = x[:, 0] if self.use_class_token else x.mean(dim = 1)

        # 3. MLP head.
        # self.mlp_head: [batch_size, num_classes]
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    batch_size = 100
    image_size = [1, 28, 28]
    patch_size = [8, 8]

    model = ViTNet(
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
