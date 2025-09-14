import os
import glob
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 데이터셋 클래스
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
        # PIL image & mask 로드
        image = Image.open(os.path.join(self.images_dir, self.img_files[idx])).convert("RGB")
        mask  = Image.open(os.path.join(self.masks_dir,  os.path.splitext(self.img_files[idx])[0] + ".png")).convert("L")
        
        # augmentation 적용
        if self.transform_image is not None:
            data = self.transform_image(image=np.array(image), mask=np.array(mask))
            image, mask = data['image'], data['mask']
        
        # mask 값 255→1 변환
        mask = torch.where(mask == 255, torch.tensor(1, dtype=mask.dtype), mask)
        mask = mask.long()
        return image, mask

# 데이터 경로 설정
base_dir = "/home/jovyan/data/l7_irish_blackratio1per"

train_source_dir = os.path.join(base_dir, "train/source")
train_label_dir = os.path.join(base_dir, "train/label")
train_image_files = sorted(glob.glob(os.path.join(train_source_dir, "*.png")))
train_image_names = [os.path.basename(p) for p in train_image_files]

val_source_dir = os.path.join(base_dir, "val/source")
val_label_dir = os.path.join(base_dir, "val/label")
val_image_files = sorted(glob.glob(os.path.join(val_source_dir, "*.png")))
val_image_names = [os.path.basename(p) for p in val_image_files]

test_source_dir = os.path.join(base_dir, "test/source")
test_label_dir = os.path.join(base_dir, "test/label")
test_image_files = sorted(glob.glob(os.path.join(test_source_dir, "*.png")))
test_image_names = [os.path.basename(p) for p in test_image_files]

train_aug = A.Compose([
    A.RandomRotate90(),  # 0/90/180/270° 회전
    A.Flip(p=0.5),       # 좌/우/상/하 뒤집기
    A.ColorJitter(       # 밝기·대비·채도·색조
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
    ),
    A.Resize(384, 384),
    A.Normalize(
        mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_aug = A.Compose([
    A.Resize(384, 384),
    A.Normalize(
        mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# 4) Dataset 생성 시 transform 변경
train_dataset = CloudDataset(
    img_files=train_image_names,
    images_dir=train_source_dir,
    masks_dir=train_label_dir,
    transform_image=train_aug,
    transform_mask=None
)
val_dataset = CloudDataset(
    img_files=val_image_names,
    images_dir=val_source_dir,
    masks_dir=val_label_dir,
    transform_image=val_aug,
    transform_mask=None
)
test_dataset = CloudDataset(
    img_files=test_image_names,
    images_dir=test_source_dir,
    masks_dir=test_label_dir,
    transform_image=val_aug,
    transform_mask=None
)

# DataLoader 구성 
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 정의 (LRASPP + MobileNetV3)
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = lraspp_mobilenet_v3_large(pretrained=False)
model.classifier.low_classifier  = nn.Conv2d(40, num_classes, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 함수
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0

    train_losses = []
    val_losses = []
    train_acc_history = []
    val_acc_history = []
    val_miou_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_pixels = 0
            cm_total = np.zeros((num_classes, num_classes), dtype=np.int32)

            phase_desc = f"{phase.capitalize()} (Epoch {epoch+1})"
            with tqdm(total=len(dataloader), desc=phase_desc, unit="batch") as pbar:
                for inputs, targets in dataloader:
                    inputs = inputs.to(device)
                    targets = targets.squeeze(1).to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        logits = outputs['out']
                        loss = criterion(logits, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == targets)
                    total_pixels += targets.numel()

                    preds_cpu = preds.detach().cpu().numpy().flatten()
                    targets_cpu = targets.detach().cpu().numpy().flatten()
                    batch_cm = confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))
                    cm_total += batch_cm

                    pbar.update(1)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / total_pixels

            intersection = np.diag(cm_total)
            union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - np.diag(cm_total)
            union = np.where(union == 0, 1, union)
            iou = intersection / union
            miou = np.mean(iou)

            if phase == 'val':
                val_losses.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                val_miou_history.append(miou)
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, mIoU: {miou:.4f}")

                if miou > best_miou:
                    best_miou = miou
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "mobilenetv3-irish-pretrain_augmentation_min-v4.pth")
                    print("Best model saved.")
            else:
                train_losses.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val mIoU: {best_miou:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# 학습 실행
num_epochs = 50
model = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=num_epochs)

# 테스트 평가 (mIoU + Accuracy)
def evaluate_test(model, dataloader, device, num_classes=2):
    model.eval()
    running_corrects = 0
    total_pixels = 0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int32)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.squeeze(1).to(device)
            outputs = model(inputs)
            logits = outputs['out']
            _, preds = torch.max(logits, 1)

            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            preds_cpu = preds.cpu().numpy().flatten()
            targets_cpu = targets.cpu().numpy().flatten()
            batch_cm = confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))
            cm_total += batch_cm

    test_accuracy = running_corrects.double() / total_pixels

    intersection = np.diag(cm_total)
    union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - np.diag(cm_total)
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    miou = np.mean(iou)

    return test_accuracy.item(), miou

test_acc, test_miou = evaluate_test(model, test_loader, device, num_classes=num_classes)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test mIoU:     {test_miou:.4f}")
