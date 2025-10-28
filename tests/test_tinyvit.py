import argparse
import json
import torch
import timm
from PIL import Image
from urllib.request import urlopen


def main():
    """
    to run test script, use the following command:
    uv run tests/test_tinyvit.py <path_to_image>
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to image")
    ap.add_argument("--model", default="tiny_vit_21m_512.dist_in22k_ft_in1k")
    ap.add_argument(
        "--topk", type=int, default=5, help="number of top predictions to show"
    )
    args = ap.parse_args()

    model = timm.create_model(args.model, pretrained=True)
    model = model.eval()

    # Load ImageNet class labels
    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    class_labels = json.loads(urlopen(imagenet_labels_url).read().decode())

    image = Image.open(args.image).convert("RGB")

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model(
        transforms(image).unsqueeze(0)
    )  # unsqueeze single image into batch of 1

    topk_probabilities, topk_class_indices = torch.topk(
        output.softmax(dim=1) * 100, k=args.topk
    )

    print(f"Top {args.topk} predictions:")
    for i in range(args.topk):
        class_idx = topk_class_indices[0][i].item()
        prob = topk_probabilities[0][i].item()
        class_name = class_labels[class_idx]
        print(f"{i + 1}. {class_name} - {prob:.2f}%")


if __name__ == "__main__":
    main()
