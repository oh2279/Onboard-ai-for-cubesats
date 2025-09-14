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

# Accuracy와 mIoU 계산
def evaluate_test(model, dataloader, device, num_classes=2):
    model.eval()
    running_corrects = torch.tensor(0, dtype=torch.float, device=device)
    total_pixels = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
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
    base_dir = "/home/jovyan/data/arirang_dataset/val"
    source_dir = os.path.join(base_dir, "source")
    label_dir = os.path.join(base_dir, "label")
    image_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    image_names = [os.path.basename(p) for p in image_files]

    # 전처리 설정
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
        T.PILToTensor()
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

    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    model = model.to(device)

    weights_path = "/home/jovyan/cvlab/MobileNetV3/cloud_detection/current/best-mobilenetv3-restylized_irish_400-finetune_head+backbone_1e5-v1.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_acc, test_miou = evaluate_test(model, dataloader, device, num_classes=num_classes)
    print(f"Val Accuracy: {test_acc:.4f}")
    print(f"Val mIoU:     {test_miou:.4f}")
