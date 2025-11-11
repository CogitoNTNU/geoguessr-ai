import argparse
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to image")  
    ap.add_argument("--texts", nargs="*", default=[
        "a street in Norway", "a street in Norway", "a desert road", "a city skyline"
    ])
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=args.texts, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits_per_image.squeeze(0)
        probs = logits.softmax(dim=-1)

    ### print stokastisk fordeling ###
    topk = min(args.topk, len(args.texts))
    idxs = torch.topk(probs, k=topk).indices.tolist()
    for i in idxs:
        print(f"{args.texts[i]}: {probs[i].item():.4f}")

if __name__ == "__main__":
    main()


#  Example run: 
#  uv run python .\tests\test_clip.py C:\Users\Romeo\Cogito\geoguessr-ai\tests\Ireland.jpg 
#  --texts "a road in Norway" "a road in Ireland" --topk [insert arg]
#
#  øk --topk for flere argumenter, default er 5
