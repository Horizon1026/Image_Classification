import torch
from model_vit import ViT

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
