import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from models import *
from src.data import get_loaders
from src.trainer import train_one_epoch, validate
from src.utils import get_metrics, save_model, print_epoch_result

def main():
    with open("configs/resnet50_imagenette.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    save_path = Path(config["save"]["checkpoint_dir"])
    save_path = save_path.joinpath(config["save"]["best_model_name"])

    train_loader, val_loader, class_names = get_loaders(config=config)

    n_class = config["model"]["n_class"]
    model = resnet50(n_class=n_class).to(device)

    train_config = config["train"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])

    epochs = train_config["epochs"]
    best_val_loss = float('inf')
    precision_metric, recall_metric, f1_metric = get_metrics(n_class=n_class, device=device)
    
    train_history = {"loss": [], "acc": []}
    val_history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        # Training ...
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        # Validating ...
        val_loss, val_acc, metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            precision_metric=precision_metric,
            recall_metric=recall_metric,
            f1_metric=f1_metric
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_path, class_names)

        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)

        val_history["loss"].append(val_loss)
        val_history["acc"].append(val_acc)

        print_epoch_result(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    metrics=metrics
                )
        
if __name__ == "__main__":
    main()






