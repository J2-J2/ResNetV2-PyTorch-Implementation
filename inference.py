import argparse
import yaml
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import random
import models
import matplotlib.pyplot as plt

def get_inference_transform(config):
    img_size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

def get_model(config):
    n_class = config["model"]["n_class"]
    model_fn = getattr(models, config["model"]["name"])
    
    return model_fn(n_class=n_class)

def load_model(config, device):
    checkpoint_dir = Path(config["save"]["checkpoint_dir"])
    best_model_name = config["save"]["best_model_name"]
    check_path = checkpoint_dir.joinpath(best_model_name)

    model = get_model(config).to(device)
    checkpoint = torch.load(check_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model, checkpoint

@torch.no_grad()
def predict_image(model, image_path, transform, class_names, device):
    ori_image = Image.open(image_path).convert("RGB")
    image = transform(ori_image)
    image = image.unsqueeze(0).to(device)

    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()

    return ori_image, idx, probs[0][idx].item()


def main():

    parser = argparse.ArgumentParser(description="J2-J2 Resnet with full preactivation")
    parser.add_argument("--image", required=False, help="추론 이미지 경로")
    args = parser.parse_args()

    with open("configs/resnet50_imagenette.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not args.image:
        image_dir = Path(config["data"]["inference_dir"])
        image_paths = list(image_dir.glob("*.JPEG"))
        image_idx = random.choice(image_paths)
        args.image = str(image_idx)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = get_inference_transform(config)

    model, checkpoint = load_model(config, device)
    class_names = checkpoint["class_names"]

    model.eval()
    image, idx, prob = predict_image(model, args.image, transform, class_names, device)

    plt.imshow(image)
    plt.title(f"Predicted Label : {class_names[idx]} | probability : {prob:.4f}")
    plt.show()


if __name__ == "__main__":
    main()