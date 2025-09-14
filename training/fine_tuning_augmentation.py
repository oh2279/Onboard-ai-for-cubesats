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
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset 정의
class CloudDataset(Dataset):
    def __init__(self, img_files, images_dir, masks_dir, transform_image=None):
        self.img_files = img_files
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_image = transform_image

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, base_name + ".png")

        # PIL 이미지 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # albumentations 적용
        if self.transform_image is not None:
            data = self.transform_image(image=np.array(image), mask=np.array(mask))
            image, mask = data['image'], data['mask']

        # mask: 255 -> 1
        mask = torch.where(mask == 255,
                           torch.tensor(1, dtype=mask.dtype),
                           mask).long()

        return image, mask

# 1) Augmentation & preprocessing 정의 (B 도메인용)
train_aug = A.Compose([
    A.RandomCrop(height=300, width=300, p=0.5),
    A.Resize(384, 384),
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    #A.Normalize(mean=[0.4943, 0.4672, 0.5249], std =[0.1129, 0.0990, 0.0898]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], additional_targets={'mask': 'mask'}, is_check_shapes=False)

val_aug = A.Compose([
    A.Resize(384, 384),
    #A.Normalize(mean=[0.4943, 0.4672, 0.5249], std =[0.1129, 0.0990, 0.0898]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], additional_targets={'mask': 'mask'}, is_check_shapes=False)

# A 도메인(Val/Test) 경로 설정
A_val_source_dir = "/home/jovyan/data/l7_irish_blackratio1per/val/source"
A_val_label_dir  = "/home/jovyan/data/l7_irish_blackratio1per/val/label"
A_test_source_dir = "/home/jovyan/data/l7_irish_blackratio1per/test/source"
A_test_label_dir  = "/home/jovyan/data/l7_irish_blackratio1per/test/label"

# A Val/ Test DataLoader
A_val_images = sorted(glob.glob(os.path.join(A_val_source_dir, "*.png")))
A_val_names  = [os.path.basename(p) for p in A_val_images]
A_val_dataset = CloudDataset(
    img_files= A_val_names,
    images_dir= A_val_source_dir,
    masks_dir= A_val_label_dir,
    transform_image= val_aug
)
A_val_loader = DataLoader(A_val_dataset, batch_size=32, shuffle=False, num_workers=4)

A_test_images = sorted(glob.glob(os.path.join(A_test_source_dir, "*.png")))
A_test_names  = [os.path.basename(p) for p in A_test_images]
A_test_dataset = CloudDataset(
    img_files= A_test_names,
    images_dir= A_test_source_dir,
    masks_dir= A_test_label_dir,
    transform_image= val_aug
)
A_test_loader = DataLoader(A_test_dataset, batch_size=32, shuffle=False, num_workers=4)

# B 도메인(스타일 변환) 경로 설정
B_source_dir = "/home/jovyan/minsu/StyleID/Sentinal2-Stylized_Irish/source"
B_label_dir  = "/home/jovyan/minsu/StyleID/Sentinal2-Stylized_Irish/label"
B_image_paths = sorted(glob.glob(os.path.join(B_source_dir, "*.png")))
B_image_names = [os.path.basename(p) for p in B_image_paths]
train_names, val_names = train_test_split(
    B_image_names, test_size=0.2, random_state=42
)

# B Train/Val DataLoader
B_train_dataset = CloudDataset(
    img_files= train_names,
    images_dir= B_source_dir,
    masks_dir= B_label_dir,
    transform_image= train_aug
)
B_train_loader = DataLoader(B_train_dataset, batch_size=32, shuffle=True, num_workers=4)

B_val_dataset = CloudDataset(
    img_files= val_names,
    images_dir= B_source_dir,
    masks_dir= B_label_dir,
    transform_image= val_aug
)
B_val_loader = DataLoader(B_val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 모델 정의 및 A로 사전 학습된 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
num_classes = 2
model = lraspp_mobilenet_v3_large(pretrained=True)
model.classifier.low_classifier  = nn.Conv2d(40, num_classes, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
model = model.to(device)

pretrained_path = "/home/jovyan/minsu/MobileNetV3/outputs/weights/pre-trained_on_irish/mobilenetv3-irish-pretrain_augmentation_min-v4.pth"
model.load_state_dict(torch.load(pretrained_path))
print("Pre-trained weights loaded from A.")

# Optimizer / Scheduler / Criterion
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-3},
    {'params': model.classifier.low_classifier.parameters(),  'lr': 1e-3},
    {'params': model.classifier.high_classifier.parameters(), 'lr': 1e-3},
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
            cm_total += confusion_matrix(targets_cpu, preds_cpu, labels=list(range(num_classes)))

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = running_corrects.double() / total_pixels
    intersection = np.diag(cm_total)
    union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union        = np.where(union == 0, 1, union)
    mIoU = np.mean(intersection / union)
    return epoch_loss, epoch_acc.item(), mIoU

# Train & Fine-tune 함수
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, A_val_loader, device, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*30)
        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
        desc = f"Train Epoch {epoch+1}"
        with tqdm(total=len(train_loader), desc=desc, unit="batch") as pbar:
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
                cm_total += confusion_matrix(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=list(range(num_classes)))
                pbar.update(1)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects.double() / total_pixels
        intersection = np.diag(cm_total)
        union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
        union        = np.where(union == 0, 1, union)
        miou_train = np.mean(intersection / union)
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | mIoU: {miou_train:.4f}")

        # B Val phase
        val_loss, val_acc, val_miou = evaluate_on_loader(model, val_loader, device)
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | mIoU: {val_miou:.4f}")
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "mobilenetv3-sentinal2_stylized_irish-fintune_1e3-augmentation-imagenet-v2.pth")
            print(">> [B Val] Best model updated & saved.")

        # A Val monitoring
        if A_val_loader is not None:
            A_loss, A_acc, A_miou = evaluate_on_loader(model, A_val_loader, device)
            print(f"A_val Loss: {A_loss:.4f} | Acc: {A_acc:.4f} | mIoU: {A_miou:.4f}")

    print(f"\nTraining completed in {(time.time()-since)//60:.0f}m {(time.time()-since)%60:.0f}s")
    print(f"Best B-val mIoU: {best_miou:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Fine-tuning 수행
num_finetune_epochs = 30
model = train_model(model, criterion, optimizer, scheduler, B_train_loader, B_val_loader, A_val_loader, device, num_finetune_epochs)

# 최종 Test 평가
def evaluate_test(model, dataloader, device):
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
            cm_total += confusion_matrix(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=list(range(num_classes)))

    avg_loss = running_loss / len(dataloader.dataset)
    avg_acc  = running_corrects.double() / total_pixels
    intersection = np.diag(cm_total)
    union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union        = np.where(union == 0, 1, union)
    avg_miou = np.mean(intersection / union)
    return avg_loss, avg_acc.item(), avg_miou

print("\n[Evaluate on A-test set]")
A_test_loss, A_test_acc, A_test_mIoU = evaluate_test(model, A_test_loader, device)
print(f"A_test Loss : {A_test_loss:.4f}")
print(f"A_test Acc  : {A_test_acc:.4f}")
print(f"A_test mIoU : {A_test_mIoU:.4f}")
