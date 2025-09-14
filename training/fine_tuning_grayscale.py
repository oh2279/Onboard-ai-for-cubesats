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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Dataset 정의 (grayscale 입력)
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

        # 이미지를 그레이스케일(1채널)로 로드
        image = Image.open(img_path).convert("L")
        mask  = Image.open(mask_path).convert("L")  # 마스크는 단일 채널

        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        # 마스크: (1, H, W) -> (H, W)로 squeeze하고, 255를 1로 변환 (클라우드 vs 논클라우드)
        mask = mask.squeeze(0)
        mask = torch.where(mask == 255, torch.tensor(1, dtype=mask.dtype), mask)
        mask = mask.long()

        return image, mask

# Transform 정의 (이미지: 1채널 → Normalize 값 1개만)
transform_image = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.5722], std=[0.2826])
])
transform_mask = T.Compose([
    T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST),
    T.PILToTensor()  # (H,W) -> (1, H, W)
])

# 데이터 경로 설정 및 DataLoader 구성

# Domain A (평가용)
A_val_source_dir = "/home/gpuadmin/data/satellite_data/equalized/irish/histogram_equal/val"
A_val_label_dir  = "/home/gpuadmin/data/satellite_data/l7_irish_blackratio1per/val/label"

A_test_source_dir = "/home/gpuadmin/data/satellite_data/equalized/irish/histogram_equal/test"
A_test_label_dir  = "/home/gpuadmin/data/satellite_data/l7_irish_blackratio1per/test/label"

A_val_images = sorted(glob.glob(os.path.join(A_val_source_dir, "*.png")))
A_val_names  = [os.path.basename(p) for p in A_val_images]
A_val_dataset = CloudDataset(
    img_files=A_val_names,
    images_dir=A_val_source_dir,
    masks_dir=A_val_label_dir,
    transform_image=transform_image,
    transform_mask=transform_mask
)
A_val_loader = DataLoader(A_val_dataset, batch_size=8, shuffle=False, num_workers=4)

A_test_images = sorted(glob.glob(os.path.join(A_test_source_dir, "*.png")))
A_test_names  = [os.path.basename(p) for p in A_test_images]
A_test_dataset = CloudDataset(
    img_files=A_test_names,
    images_dir=A_test_source_dir,
    masks_dir=A_test_label_dir,
    transform_image=transform_image,
    transform_mask=transform_mask
)
A_test_loader = DataLoader(A_test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Domain B (스타일 변환 데이터 → 파인튜닝용)
B_source_dir = "/home/gpuadmin/satellite/minsu/StyleID/stylized_irish_histequal_400/source"
B_label_dir  = "/home/gpuadmin/satellite/minsu/StyleID/stylized_irish_histequal_400/label"

B_image_paths = sorted(glob.glob(os.path.join(B_source_dir, "*.png")))
B_image_names = [os.path.basename(p) for p in B_image_paths]

# 전체 이미지 중 80%는 훈련, 20%는 검증으로 분할
train_names, val_names = train_test_split(B_image_names, test_size=0.2, random_state=42)

B_train_dataset = CloudDataset(
    img_files=train_names,
    images_dir=B_source_dir,
    masks_dir=B_label_dir,
    transform_image=transform_image,
    transform_mask=transform_mask
)
B_train_loader = DataLoader(B_train_dataset, batch_size=8, shuffle=True, num_workers=4)

B_val_dataset = CloudDataset(
    img_files=val_names,
    images_dir=B_source_dir,
    masks_dir=B_label_dir,
    transform_image=transform_image,
    transform_mask=transform_mask
)
B_val_loader = DataLoader(B_val_dataset, batch_size=8, shuffle=False, num_workers=4)

# 모델 정의 및 사전학습 가중치 로드 (A 도메인 기반)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

num_classes = 2
model = lraspp_mobilenet_v3_large(pretrained=False)
model.classifier.low_classifier  = nn.Conv2d(40, num_classes, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
# Backbone의 첫 번째 conv layer 수정 전에 모델을 GPU에 올림
model = model.to(device)

# checkpoint 로드 전에 backbone의 첫 번째 conv layer를 1채널 입력에 맞게 변경
first_key = list(model.backbone._modules.keys())[0]
conv_norm_act = model.backbone._modules[first_key]  # Sequential([Conv2d, BatchNorm2d, Activation])
old_conv = conv_norm_act[0]
new_conv = nn.Conv2d(
    in_channels=1,  # 1채널 입력으로 수정
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=(old_conv.bias is not None)
)
with torch.no_grad():
    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
# GPU에 있는 모델과 일치하도록 new_conv도 GPU로 이동
conv_norm_act[0] = new_conv.to(device)
# ----------------------------------------------------------------

# 이제 checkpoint 로드 (현재 모델의 첫 번째 conv layer shape는 [16, 1, 3, 3]로 변경됨)
pretrained_path = "/home/gpuadmin/satellite/minsu/mobilenetv3-hist_equal-pretrain-v1.pth"
model.load_state_dict(torch.load(pretrained_path))
print("Pre-trained weights loaded from A.")

# Backbone의 파라미터 동결 – 필요에 따라 주석 해제
# for name, param in model.backbone.named_parameters():
#     param.requires_grad = False

# Optimizer, Scheduler, Loss 함수 설정
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.low_classifier.parameters(),  'lr': 1e-5},
    {'params': model.classifier.high_classifier.parameters(), 'lr': 1e-5},
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 평가 함수 정의
def evaluate_on_loader(model, dataloader, device, num_classes=2):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_pixels = 0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits  = outputs['out']
            loss    = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            preds_cpu   = preds.detach().cpu().numpy().flatten()
            targets_cpu = targets.detach().cpu().numpy().flatten()
            batch_cm = confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))
            cm_total += batch_cm

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = running_corrects.double() / total_pixels
    intersection = np.diag(cm_total)
    union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    mIoU = np.mean(iou)
    return epoch_loss, epoch_acc.item(), mIoU

# 학습 함수 정의 (B 도메인 데이터로 파인튜닝, A 도메인 성능 모니터링 포함)
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, A_val_loader, device, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0  # B 도메인 검증 기준 최적 mIoU

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # 8-1. Training (B_train)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

        train_desc = f"Train Epoch {epoch+1}"
        with tqdm(total=len(train_loader), desc=train_desc, unit="batch") as pbar:
            for inputs, targets in train_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits  = outputs['out']
                loss    = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                running_corrects += torch.sum(preds == targets)
                total_pixels += targets.numel()

                preds_cpu   = preds.detach().cpu().numpy().flatten()
                targets_cpu = targets.detach().cpu().numpy().flatten()
                batch_cm = confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))
                cm_total += batch_cm

                pbar.update(1)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects.double() / total_pixels
        intersection = np.diag(cm_total)
        union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
        union = np.where(union == 0, 1, union)
        iou = intersection / union
        miou_train = np.mean(iou)

        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | mIoU: {miou_train:.4f}")

        # Validation on B domain
        val_loss, val_acc, val_miou = evaluate_on_loader(model, val_loader, device, num_classes)
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | mIoU: {val_miou:.4f}")

        # 베스트 모델 갱신 (B 도메인 기준)
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "mobilenetv3-stylized_irish_histequal-finetune_1e5-v1.pth")
            print(">> [B Val] Best model updated & saved.")

        # A domain validation (모니터링)
        if A_val_loader is not None:
            A_val_loss, A_val_acc, A_val_miou = evaluate_on_loader(model, A_val_loader, device, num_classes)
            print(f"A_val Loss: {A_val_loss:.4f} | Acc: {A_val_acc:.4f} | mIoU: {A_val_miou:.4f}")

    time_elapsed = time.time() - since
    print(f"\nTraining finished in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best B-val mIoU: {best_miou:.4f}")

    model.load_state_dict(best_model_wts)
    return model

# 파인튜닝 실행 (B 도메인 train/val, A 도메인 모니터링)
num_finetune_epochs = 30
model = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=B_train_loader,
    val_loader=B_val_loader,
    A_val_loader=A_val_loader,
    device=device,
    num_epochs=num_finetune_epochs
)

# 테스트 평가 함수 정의 (A_test)
def evaluate_test(model, dataloader, device, num_classes=2):
    model.eval()
    running_corrects = 0
    total_pixels = 0
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            logits  = outputs['out']
            loss    = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            preds_cpu   = preds.detach().cpu().numpy().flatten()
            targets_cpu = targets.detach().cpu().numpy().flatten()
            batch_cm = confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))
            cm_total += batch_cm

    avg_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / total_pixels

    intersection = np.diag(cm_total)
    union = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    miou = np.mean(iou)
    return avg_loss, test_acc.item(), miou

# A 도메인 테스트셋 평가
print("\n[Evaluate on A-test set]")
A_test_loss, A_test_acc, A_test_mIoU = evaluate_test(model, A_test_loader, device, num_classes)
print(f"A_test Loss : {A_test_loss:.4f}")
print(f"A_test Acc  : {A_test_acc:.4f}")
print(f"A_test mIoU : {A_test_mIoU:.4f}")
