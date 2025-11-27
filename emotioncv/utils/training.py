
from typing import Tuple

import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, device, dataloader: DataLoader, optimizer, criterion) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def evaluate(model, device, dataloader: DataLoader, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy
