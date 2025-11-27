
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from emotioncv.data.datasets import FaceConfig, FaceEmotionDataset
from emotioncv.models.face_cnn import EmotionNet
from emotioncv.utils.training import train_one_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "face_data"
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    cfg = FaceConfig(root_dir=str(data_root), img_size=48)

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
    ])

    dataset = FaceEmotionDataset(cfg, transform=transform)
    if len(dataset) == 0:
        print("[ERROR] No face images found. Please populate face_data/ with emotion folders.")
        return

    # simple train/val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = EmotionNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        print(f"Epoch {epoch}/{num_epochs} "
              f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoints_dir / "face_emotion_best.pth"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            print(f"[INFO] Saved best model to {ckpt_path}")

    print(f"[DONE] Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
