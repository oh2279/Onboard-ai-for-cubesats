# /home/gpuadmin/data/satellite_data/Sentinel2/image  위 경로의 데이터 읽어와서 저장된 checkpoint로 추론하는 코드
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

    
class Sentinel2Dataset(Dataset):
    def __init__(self, root_dir, transform_image=None, transform_mask=None):
        self.root_dir = root_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        
        raw_files = sorted(glob.glob(os.path.join(self.root_dir+"/source", "*_Raw.tif")))
        label_files = sorted(glob.glob(os.path.join(self.root_dir+"/label", "*_CloudLabel.tif")))
        assert len(raw_files) == len(label_files), "Raw 파일과 label 파일의 개수가 동일해야 합니다."
        
        self.raw_files = raw_files
        self.label_files = label_files

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]
        label_path = self.label_files[idx]
        

        image = Image.open(raw_path)
        image = image.convert("RGB")
        mask = Image.open(label_path).convert("L")

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # mask의 경우, 255를 1로 변환하는 코드
        mask = mask.squeeze(0)
        mask = torch.where(mask == 255, torch.tensor(1, dtype=mask.dtype), mask)
        mask = mask.long()
        return image, mask


def calculate_accuracy_and_miou(preds, targets, num_classes):
    
    all_preds = torch.cat(preds).view(-1)
    all_targets = torch.cat(targets).view(-1)
    cm = confusion_matrix(all_targets.numpy(), all_preds.numpy(), labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    miou = np.mean(iou)
    
    return miou

        
# 평가 함수를 그대로 사용 (정확도, mIoU 계산)
def evaluate_test(model, dataloader, device, num_classes=2):
    model.eval()
    running_corrects = torch.tensor(0, dtype=torch.float, device=device)
    total_pixels = 0
    all_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # targets는 채널 차원이 있는 상태이므로 squeeze
            targets = targets.squeeze(1).to(device)
            outputs = model(inputs)
            logits = outputs['out']
            #logits = torch.sigmoid(logits)
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == targets)
            total_pixels += targets.numel()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())

        test_accuracy = running_corrects.double() / total_pixels
        miou = calculate_accuracy_and_miou(all_preds, all_targets, num_classes)
        #show_result(all_inputs, all_preds, all_targets)

    return test_accuracy.item(), miou

def denormalize(tensor, mean, std):
    """
    tensor: torch.Tensor (3, H, W), 정규화된 이미지
    mean, std: 리스트 or Tensor of shape (3,)
    """
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor * std + mean)

def show_result(all_inputs, all_preds, all_targets):
    save_dir = "/home/jovyan/minsu/results"
    os.makedirs(save_dir, exist_ok=True)

    for idx, (input_tensor, pred_tensor, target_tensor) in enumerate(zip(all_inputs, all_preds, all_targets)):
        input_denorm = denormalize(input_tensor, mean, std)
        input = input_denorm.squeeze(0).numpy().transpose(1, 2, 0)
        
        pred = pred_tensor.squeeze(0).numpy().astype(np.uint8)
        target = target_tensor.squeeze(0).numpy().astype(np.uint8)

        #input = (input - input.min()) / (input.max() - input.min())
        #input = input.astype(np.uint64)
        overlay = np.zeros_like(input)
        overlay2 = np.zeros_like(input)
        # 빨강: FN (GT는 1인데 예측 못한 부분)
        overlay[(target == 1) & (pred == 0)] = (255, 0, 0)

        # 파랑: FP (예측은 했는데 GT에는 없는 부분)
        overlay[(target == 0) & (pred == 1)] = (0, 0, 255)

        # 초록: TP (정확히 맞춘 부분)
        overlay[(target == 1) & (pred == 1)] = (0, 255, 0)
        alpha = 0.4 # 투명도

        overlay2[pred == 1] = (255, 255, 0)
        seg_map1 = ((1 - alpha) * input + alpha * overlay)
        seg_map2 = ((1 - alpha) * input + alpha * overlay2)
        
        titles = ['Input', 'Prediction', 'Ground Truth', 'Segmentation Map 1', 'Segmentation Map 2']
        images = [input, pred, target, seg_map1, seg_map2]
        cmaps = [None, 'gray', 'gray', None, None]  # 원하는 colormap 설정

        plt.figure(figsize=(15, 4))

        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i], cmap=cmaps[i])
            plt.title(titles[i], fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"segmentation_map_{idx}.png"), dpi=300)
        plt.close()
        
        
mean = [0.4943, 0.4672, 0.5249]
std = [0.1129, 0.0990, 0.0898]

if __name__ == "__main__":
    # 새로운 Sentinel2 데이터셋 경로 설정
    root_dir = "/home/jovyan/minsu/StyleID/Sentinal2"

    # 전처리 설정 (기존과 동일하게 384x384로 리사이즈)
    transform_image = T.Compose([
        T.Resize((384, 384)),   # 필요 시 원본 사이즈에 맞게 변경하거나 패치 단위 추론 고려
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]
        )
    ])
    transform_mask = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor()
    ])

    dataset = Sentinel2Dataset(
        root_dir,
        transform_image=transform_image,
        transform_mask=transform_mask
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 구성: 기존과 동일하게 LR-ASPP MobileNetV3 Large 사용
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    model = model.to(device)

    weights_path = "/home/jovyan/minsu/MobileNetV3/mobilenetv3-sentinal2_stylized_irish-fintune-optuna-v1.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_acc, test_miou = evaluate_test(model, dataloader, device, num_classes=num_classes)

    print(f"Val Accuracy: {test_acc:.4f}")
    print(f"Val mIoU:     {test_miou:.4f}")