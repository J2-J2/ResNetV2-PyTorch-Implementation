# src/trainer.py

import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(
        train_loader,
        desc=f"[Epoch {epoch + 1}] Train",
        leave=False
    )

    for inputs, labels in train_pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels).item()

        train_pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{(correct / total) * 100:.2f}%"
        )

    train_loss = train_loss / total
    train_acc = (correct / total) * 100

    return train_loss, train_acc


@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch,
    precision_metric=None,
    recall_metric=None,
    f1_metric=None,
):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_pbar = tqdm(
        val_loader,
        desc=f"[Epoch {epoch + 1}] Val",
        leave=False
    )

    for inputs, labels in val_pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)

        predicted = torch.argmax(outputs, dim=1)
        val_total += labels.size(0)
        val_correct += torch.sum(predicted == labels).item()

        if precision_metric is not None:
            precision_metric.update(predicted, labels)
        if recall_metric is not None:
            recall_metric.update(predicted, labels)
        if f1_metric is not None:
            f1_metric.update(predicted, labels)

        val_pbar.set_postfix(
            val_loss=f"{loss.item():.4f}"
        )

    val_loss = val_loss / val_total
    val_acc = (val_correct / val_total) * 100

    metrics = {}

    if precision_metric is not None:
        metrics["precision"] = precision_metric.compute().item()
        precision_metric.reset()

    if recall_metric is not None:
        metrics["recall"] = recall_metric.compute().item()
        recall_metric.reset()

    if f1_metric is not None:
        metrics["f1"] = f1_metric.compute().item()
        f1_metric.reset()

    return val_loss, val_acc, metrics