"""
# tsne_compare.py
# 하드코딩 + tqdm + MobileNetV3 (ImageNet pretrained) + 모든 데이터 포함

import os
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 하드코딩 설정 ===========
dirA = "/home/jovyan/data/arirang_dataset/val/source"
dirB = "/home/jovyan/minsu/StyleID/restylized_irish_400/source"
perplexity = 30
learning_rate = 200
n_iter = 1000
output = "/home/jovyan/minsu/tsne_images.png"
# ====================================

def load_image_paths(dir_path):
    exts = ('.tif', '.png', '.jpg', '.jpeg')
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(exts)]

def extract_features(image_paths, model, device, transform):
    features = []
    model.eval()
    with torch.no_grad():
        for path in tqdm(image_paths, desc="특징 추출", unit="image", ncols=80):
            img = Image.open(path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            feat = model(x)
            features.append(feat.cpu().numpy().squeeze())
    return np.array(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("이미지 경로 로드 중...")
    imgs_A = load_image_paths(dirA)
    imgs_B = load_image_paths(dirB)
    if not imgs_A or not imgs_B:
        raise ValueError('디렉터리에 이미지 파일이 없습니다.')

    print(f"Dataset A: {len(imgs_A)}장, Dataset B: {len(imgs_B)}장 이미지 로드 완료.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898])
    ])

    print("MobileNetV3 모델 로드 중 (ImageNet pretrained)...")
    backbone = models.mobilenet_v3_large(pretrained=True)

    model = torch.nn.Sequential(
        backbone.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    ).to(device)

    print("Dataset A 특징 추출...")
    feats_A = extract_features(imgs_A, model, device, transform)
    print("Dataset B 특징 추출...")
    feats_B = extract_features(imgs_B, model, device, transform)

    print("t-SNE 계산 중...")
    X = np.vstack([feats_A, feats_B])
    labels = np.array([0] * len(feats_A) + [1] * len(feats_B))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42
    )
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_emb[labels == 0, 0], X_emb[labels == 0, 1],
                label='Target Dataset', alpha=0.6, s=20)
    plt.scatter(X_emb[labels == 1, 0], X_emb[labels == 1, 1],
                label='Target-Stylized Source Dataset', alpha=0.6, s=20)
    plt.legend()
    plt.title('Feature Embedding Comparison')
    plt.xlabel('Embedding Dim 1')
    plt.ylabel('Embedding Dim 2')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"t-SNE 이미지 플롯이 '{output}'로 저장되었습니다.")

if __name__ == '__main__':
    main()


"""
# tsne_compare.py
# 하드코딩 + tqdm + MobileNetV3 (ImageNet pretrained) + 모든 데이터 포함

import os
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 하드코딩 설정 ===========
dirA = "/home/jovyan/data/arirang_dataset/val/source"
dirB = "/home/jovyan/data/l7_irish_blackratio1per/train/source"
perplexity = 30
learning_rate = 200
n_iter = 1000
output = "/home/jovyan/minsu/tsne_images.png"
# ====================================

def load_image_paths(dir_path):
    exts = ('.tif', '.png', '.jpg', '.jpeg')
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(exts)]

def extract_features(image_paths, model, device, transform):
    features = []
    model.eval()
    with torch.no_grad():
        for path in tqdm(image_paths, desc="특징 추출", unit="image", ncols=80):
            img = Image.open(path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            feat = model(x)
            features.append(feat.cpu().numpy().squeeze())
    return np.array(features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("이미지 경로 로드 중...")
    imgs_A = load_image_paths(dirA)
    imgs_B = load_image_paths(dirB)
    if not imgs_A or not imgs_B:
        raise ValueError('디렉터리에 이미지 파일이 없습니다.')

    n = min(len(imgs_A), len(imgs_B))
    imgs_A = imgs_A[:n]
    imgs_B = imgs_B[:n]
    print(f"각 데이터셋에서 {n}장씩 샘플링: Dataset A/B = {n}/{n}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
        #transforms.Normalize(mean=[0.4696, 0.5026, 0.4939], std=[0.3604, 0.3468, 0.3566])
    ])

    print("MobileNetV3 모델 로드 중 (ImageNet pretrained)...")
    backbone = models.mobilenet_v3_large(pretrained=True)  # 외부 가중치 제거, ImageNet pretrained 사용

    model = torch.nn.Sequential(
        backbone.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    ).to(device)

    print("Dataset A 특징 추출...")
    feats_A = extract_features(imgs_A, model, device, transform)
    print("Dataset B 특징 추출...")
    feats_B = extract_features(imgs_B, model, device, transform)

    print("t-SNE 계산 중...")
    X = np.vstack([feats_A, feats_B])
    labels = np.array([0] * len(feats_A) + [1] * len(feats_B))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42
    )
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_emb[labels == 0, 0], X_emb[labels == 0, 1],
                label='Target Dataset', alpha=0.6, s=20)
    plt.scatter(X_emb[labels == 1, 0], X_emb[labels == 1, 1],
                label='Source Dataset', alpha=0.6, s=20)
    plt.legend()
    plt.title('Feature Embedding Comparison')
    plt.xlabel('Embedding Dim 1')
    plt.ylabel('Embedding Dim 2')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"t-SNE 이미지 플롯이 '{output}'로 저장되었습니다.")

if __name__ == '__main__':
    main()
