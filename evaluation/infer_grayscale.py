import os
import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

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
        mask = Image.open(mask_path).convert("L")

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # 마스크: (1, H, W) -> (H, W)로 squeeze하고, 255를 1로 변환 (클라우드 vs 논클라우드)
        mask = mask.squeeze(0)
        mask = torch.where(mask == 255, torch.tensor(1, dtype=mask.dtype), mask)
        mask = mask.long()
        return image, mask

# Accuracy와 mIoU 계산 함수 (변경 없음)
def evaluate_test(model, dataloader, device, num_classes=2):
    model.eval()
    running_corrects = torch.tensor(0, dtype=torch.float, device=device)
    total_pixels = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # targets: (1, H, W) -> (H, W)
            targets = targets.squeeze(1).to(device)
            outputs = model(inputs)
            logits = outputs['out']
            _, preds = torch.max(logits, 1)

            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    test_accuracy = running_corrects.double() / total_pixels

    all_preds = torch.cat(all_preds).view(-1)
    all_targets = torch.cat(all_targets).view(-1)
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    miou = np.mean(iou)

    return test_accuracy.item(), miou

if __name__ == "__main__":
    # 데이터 경로 설정 (여기서는 val 데이터셋 예시)
    base_dir = "/home/gpuadmin/data/satellite_data/equalized/arirang/histogram_equal/val"
    source_dir = os.path.join(base_dir, "")
    label_dir = "/home/gpuadmin/data/satellite_data/arirang_dataset/val/label"
    image_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    image_names = [os.path.basename(p) for p in image_files]

    # 전처리 설정 (1채널에 맞게 Normalize)
    transform_image = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        # 1채널 grayscale이므로 1개 값만 사용 (데이터셋 통계에 따라 수정 필요)
        T.Normalize(mean=[0.5722], std=[0.2826])
    ])
    transform_mask = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor()  # (H, W) -> (1, H, W)
    ])

    dataset = CloudDataset(
        img_files=image_names,
        images_dir=source_dir,
        masks_dir=label_dir,
        transform_image=transform_image,
        transform_mask=transform_mask
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 생성 (pretrained=False로 생성한 후 checkpoint를 로드할 예정)
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    model = model.to(device)

    # backbone의 첫 번째 conv layer를 1채널 입력에 맞게 수정 (checkpoint 로드 전에 수행)
    first_key = list(model.backbone._modules.keys())[0]
    conv_norm_act = model.backbone._modules[first_key]  # 일반적으로 Sequential([Conv2d, BatchNorm2d, Activation])
    old_conv = conv_norm_act[0]
    new_conv = nn.Conv2d(
        in_channels=1,  # 1채널 입력
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    # 새 conv layer를 GPU로 이동한 후 대체
    conv_norm_act[0] = new_conv.to(device)

    # checkpoint 가중치 로드 (이미 1채널 입력용으로 저장된 checkpoint)
    weights_path = "/home/gpuadmin/satellite/minsu/mobilenetv3-stylized_irish_histequal-finetune_1e5-v1.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_acc, test_miou = evaluate_test(model, dataloader, device, num_classes=num_classes)
    print(f"Val Accuracy: {test_acc:.4f}")
    print(f"Val mIoU:     {test_miou:.4f}")
