from models import *
import yaml
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
from torchinfo import summary
from data import get_loaders
import torch.optim

with open("configs/resnet50_imagenette.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = "gpu" if torch.cuda.is_avilable() else "cpu"

train_loader, val_loader = get_loaders(config=config)

n_class = config["model"]["n_classes"]
model = resnet50(n_class =n_class)

train_config = config["train"]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

epochs = train_config["epochs"]



