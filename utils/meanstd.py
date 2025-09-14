import os
import glob
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

# 데이터셋 경로
dataset_path = "/home/jovyan/data/95_cloud/train/source"
image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.png")))

# 전처리: Resize 등 필요에 따라 추가 가능
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),  # 이미지를 [0,1] 범위의 텐서로 변환, (C, H, W) 형태
])

# 각 이미지의 채널별 평균, std를 저장할 리스트
means = []
stds = []

for img_path in tqdm(image_paths, desc="Processing images"):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image)  # (3, H, W)
    
    # 이미지의 각 채널별 평균과 std 계산
    means.append(tensor.mean(dim=[1, 2]))
    stds.append(tensor.std(dim=[1, 2]))

# 텐서로 변환 후 전체 이미지에 대해 평균 및 표준편차 계산
means = torch.stack(means)
stds = torch.stack(stds)

dataset_mean = means.mean(dim=0)
dataset_std = stds.mean(dim=0)

print("Dataset Mean:", dataset_mean)
print("Dataset Std: ", dataset_std)
