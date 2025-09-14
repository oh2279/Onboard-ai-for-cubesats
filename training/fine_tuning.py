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

# Dataset 정의
class CloudDataset(Dataset):
    def __init__(self, img_files, images_dir, masks_dir,transform_image=None, transform_mask=None):
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
        mask = Image.open(mask_path).convert("L")  # 단일 채널(클라우드 vs 논클라우드)

        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        # 마스크 값 0, 255 -> 0/1 클래스로 변환
        mask = mask.squeeze(0)  # (1,H,W) -> (H,W)
        mask = torch.where(mask == 255,torch.tensor(1, dtype=mask.dtype),mask)
        mask = mask.long()  # CrossEntropyLoss를 위해 long 텐서로 변환

        return image, mask

# Transform 정의
transform_image = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    # imagenet 
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # irish_blackratio1per 
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898])

    # clahe_irish
    #T.Normalize(mean=[0.5058, 0.4787, 0.5364], std=[0.1434, 0.1302, 0.1196])

    # ycrcb_irish
    #T.Normalize(mean=[0.5827, 0.5576, 0.6135], std=[0.2888, 0.2803, 0.2661])
    
    # 95-cloud 
    #T.Normalize(mean=[0.2189, 0.2164, 0.2303], std=[0.0459, 0.0417, 0.0409])
])

transform_mask = T.Compose([
    T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST),
    T.PILToTensor()  # (H,W) -> (1,H,W)
])

# A 데이터 경로 설정 (Val/Test 용도)
# A로 이미 학습된 모델이 있으니, 학습 중 A 성능 모니터링 or 최종평가를 위해 로더 구성
A_val_source_dir = "/home/jovyan/data/l7_irish_blackratio1per/val/source"
A_val_label_dir  = "/home/jovyan/data/l7_irish_blackratio1per/val/label"

A_test_source_dir = "/home/jovyan/data/l7_irish_blackratio1per/test/source"
A_test_label_dir  = "/home/jovyan/data/l7_irish_blackratio1per/test/label"

# A val loader
A_val_images = sorted(glob.glob(os.path.join(A_val_source_dir, "*.png")))
A_val_names  = [os.path.basename(p) for p in A_val_images]

A_val_dataset = CloudDataset(
    img_files       = A_val_names,
    images_dir      = A_val_source_dir,
    masks_dir       = A_val_label_dir,
    transform_image = transform_image,
    transform_mask  = transform_mask
)
A_val_loader = DataLoader(A_val_dataset, batch_size=32,shuffle=False, num_workers=4)

# A test loader
A_test_images = sorted(glob.glob(os.path.join(A_test_source_dir, "*.png")))
A_test_names  = [os.path.basename(p) for p in A_test_images]

A_test_dataset = CloudDataset(
    img_files       = A_test_names,
    images_dir      = A_test_source_dir,
    masks_dir       = A_test_label_dir,
    transform_image = transform_image,
    transform_mask  = transform_mask
)
A_test_loader = DataLoader(A_test_dataset, batch_size=32,shuffle=False, num_workers=4)

# B(스타일 변환) 데이터 경로 설정
B_source_dir = "/home/jovyan/minsu/StyleID/restylized_irish_400/source"
B_label_dir  = "/home/jovyan/minsu/StyleID/restylized_irish_400/label"

B_image_paths = sorted(glob.glob(os.path.join(B_source_dir, "*.png")))
B_image_names = [os.path.basename(p) for p in B_image_paths]

# 80%: Train, 20%: Validation
train_names, val_names = train_test_split(
    B_image_names, test_size=0.2, random_state=42
)

# B train dataset/loader
B_train_dataset = CloudDataset(
    img_files       = train_names,
    images_dir      = B_source_dir,
    masks_dir       = B_label_dir,
    transform_image = transform_image,
    transform_mask  = transform_mask
)
B_train_loader = DataLoader(B_train_dataset, batch_size=32,shuffle=True, num_workers=4)

# B val dataset/loader
B_val_dataset = CloudDataset(
    img_files       = val_names,
    images_dir      = B_source_dir,
    masks_dir       = B_label_dir,
    transform_image = transform_image,
    transform_mask  = transform_mask
)
B_val_loader = DataLoader(B_val_dataset, batch_size=32,shuffle=False, num_workers=4)

# 모델 정의 및 사전 학습 가중치 로드
# A로 사전 학습된 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

num_classes = 2
model = lraspp_mobilenet_v3_large(pretrained=False)
model.classifier.low_classifier  = nn.Conv2d(40,  num_classes, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
model = model.to(device)

# A로 학습된 모델 가중치
pretrained_path = "/home/jovyan/minsu/MobileNetV3/outputs/weights/pre-trained_on_irish/mobilenetv3-irish-pretrain-v1.pth"
model.load_state_dict(torch.load(pretrained_path))
print("Pre-trained weights loaded from A.")

# Backbone(혹은 필요 레이어) 동결
# A에서 학습된 특징 
for name, param in model.backbone.named_parameters():
    param.requires_grad = False

# Optimizer / Scheduler 정의
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 0.00639},
    {'params': model.classifier.low_classifier.parameters(),  'lr': 0.00056},
    {'params': model.classifier.high_classifier.parameters(), 'lr': 0.00056},
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1938)

criterion = nn.CrossEntropyLoss()

# 학습 및 검증 함수
# B의 train/val 세트로 훈련 & best model 갱신
# 동시에 A_val 성능도 로그에
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
            batch_cm    = confusion_matrix(targets_cpu,preds_cpu,labels=list(range(num_classes)))
            cm_total += batch_cm

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc  = running_corrects.double() / total_pixels

    # IoU 계산
    intersection = np.diag(cm_total)
    union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union        = np.where(union == 0, 1, union)  # 0 방지
    iou  = intersection / union
    mIoU = np.mean(iou)

    return epoch_loss, epoch_acc.item(), mIoU


def train_model(model, 
                criterion,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                A_val_loader,   # A의 val 성능 모니터링용
                device,
                num_epochs=10):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0.0  # B의 val 기준

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # 1) Train (B train)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

        train_desc = f"Train Epoch {epoch+1}"
        with tqdm(total=len(train_loader), desc=train_desc, unit="batch", miniters=10) as pbar:
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
                batch_cm    = confusion_matrix(targets_cpu,preds_cpu,labels=list(range(num_classes)))
                cm_total += batch_cm

                pbar.update(1)

        scheduler.step()  # 한 epoch 끝날 때 scheduler 갱신

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects.double() / total_pixels
        intersection = np.diag(cm_total)
        union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
        union        = np.where(union == 0, 1, union)
        iou  = intersection / union
        miou_train = np.mean(iou)

        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | mIoU: {miou_train:.4f}")

        # 2) Validation (B val) → best model 판단용
        val_loss, val_acc, val_miou = evaluate_on_loader(model, val_loader, device, num_classes)
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | mIoU: {val_miou:.4f}")

        # 베스트 모델 갱신 (B val 기준)
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),"mobilenetv3-arirang_stylized_irish-fintune-optuna-v1.pth")
            print(">> [B Val] Best model updated & saved.")

        # 3) A_val 성능 모니터링
        # 학습 목표가 B 도메인이지만, A 성능이 어떻게 변하는지
        if A_val_loader is not None:
            A_val_loss, A_val_acc, A_val_miou = evaluate_on_loader(
                model, A_val_loader, device, num_classes
            )
            print(f"A_val Loss: {A_val_loss:.4f} | Acc: {A_val_acc:.4f} | mIoU: {A_val_miou:.4f}")

    time_elapsed = time.time() - since
    print(f"\nTraining finished in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best B-val mIoU: {best_miou:.4f}")

    # 학습 종료 후, best 모델 로드
    model.load_state_dict(best_model_wts)
    return model

# 파인튜닝 수행
num_finetune_epochs = 30
model = train_model(
    model            = model,
    criterion        = criterion,
    optimizer        = optimizer,
    scheduler        = scheduler,
    train_loader     = B_train_loader,  # B 훈련
    val_loader       = B_val_loader,    # B 검증 (베스트 선정)
    A_val_loader     = A_val_loader,    # 매 epoch마다 A_val 모니터링
    device           = device,
    num_epochs       = num_finetune_epochs
)

# 최종 Test 단계
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
            batch_cm    = confusion_matrix(targets_cpu,preds_cpu,labels=list(range(num_classes)))
            cm_total += batch_cm

    avg_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / total_pixels

    intersection = np.diag(cm_total)
    union        = cm_total.sum(axis=1) + cm_total.sum(axis=0) - intersection
    union        = np.where(union == 0, 1, union)
    iou  = intersection / union
    miou = np.mean(iou)

    return avg_loss, test_acc.item(), miou


print("\n[Evaluate on A-test set]")
A_test_loss, A_test_acc, A_test_mIoU = evaluate_test(model, A_test_loader, device, num_classes)
print(f"A_test Loss : {A_test_loss:.4f}")
print(f"A_test Acc  : {A_test_acc:.4f}")
print(f"A_test mIoU : {A_test_mIoU:.4f}")
