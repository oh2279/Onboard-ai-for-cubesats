import os
import glob
import time
import copy
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

import optuna

# --------------------------
# Dataset
# --------------------------
class CloudDataset(Dataset):
    def __init__(self, img_files, images_dir, masks_dir, transform_image=None, transform_mask=None):
        self.img_files = img_files
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, base_name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = mask.squeeze(0)
        mask = torch.where(mask == 255, torch.tensor(1, dtype=mask.dtype), mask)
        mask = mask.long()
        return image, mask

# --------------------------
# Evaluation
# --------------------------
def evaluate_on_loader(model, dataloader, device, num_classes=2):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_pixels = 0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits = outputs['out']
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            preds_cpu = preds.cpu().numpy().flatten()
            targets_cpu = targets.cpu().numpy().flatten()
            cm_total += confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / total_pixels

    intersection = np.diag(cm_total)
    union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    mIoU = np.mean(iou)
    return epoch_loss, epoch_acc.item(), mIoU

# --------------------------
# Model Creator
# --------------------------
def get_model(num_classes, pretrained_path):
    model = lraspp_mobilenet_v3_large(weights=None)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    model.load_state_dict(torch.load(pretrained_path))
    return model

# --------------------------
# Training
# --------------------------
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs['out']
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        _, _, val_miou = evaluate_on_loader(model, val_loader, device)

        if val_miou > best_miou:
            best_miou = val_miou
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return best_miou

# --------------------------
# Optuna Objective
# --------------------------
def objective(trial):
    # 하이퍼파라미터 샘플링
    # 기존 best 기준 좁은 범위 재탐색
    lr_head = trial.suggest_float("lr_head", 0.0005, 0.00065, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 0.0055, 0.0075, log=True)

    step_size = trial.suggest_int("step_size", 8, 10)
    gamma = trial.suggest_float("gamma", 0.15, 0.25)

    norm_type = trial.suggest_categorical("norm_type", ["irish"])

    # Normalize 설정
    if norm_type == "imagenet":
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif norm_type == "irish":
        norm = T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898])

    transform_image = T.Compose([T.Resize((384, 384)), T.ToTensor(), norm])
    transform_mask = T.Compose([T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST), T.PILToTensor()])

    # 데이터 로딩
    source_dir = "/home/jovyan/minsu/StyleID/restylized_irish_400/source"
    label_dir = "/home/jovyan/minsu/StyleID/restylized_irish_400/label"
    image_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    image_names = [os.path.basename(f) for f in image_files]
    train_names, val_names = train_test_split(image_names, test_size=0.2, random_state=42)

    train_dataset = CloudDataset(train_names, source_dir, label_dir, transform_image, transform_mask)
    val_dataset = CloudDataset(val_names, source_dir, label_dir, transform_image, transform_mask)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 모델, 옵티마이저, 스케줄러
    model = get_model(num_classes=2, pretrained_path="/home/jovyan/minsu/MobileNetV3/outputs/weights/pre-trained_on_irish/mobilenetv3-irish-pretrain-v1.pth")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.classifier.low_classifier.parameters(), 'lr': lr_head},
        {'params': model.classifier.high_classifier.parameters(), 'lr': lr_head},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_miou = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=10)
    return best_miou

# --------------------------
# 실행
# --------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    print(f"  mIoU: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
