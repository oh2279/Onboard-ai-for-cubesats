import os
import time
import random
import glob
import tarfile
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import cv2

"""
예시 코드
--weights_path: 실제 best.pth 위치
--image_dir: 촬영한 이미지가 저장된 위치
--save_dir: 결과를 저장할 위치(없앨 수도 있을 것 같습니다.)
--tar_name: tar 이름 설정
--name: 사용할 이미지 그냥 이름만 넣어주시면 됩니다.
python package.py \
  --weights_path /home/cvlab/Desktop/2025/quantization/cloud_detection/current/best-mobilenetv3-restylized_irish_400-finetune_head+backbone_1e5-v1.pth \
  --image_dir /home/cvlab/Desktop/2025/quantization/cloud_detection/raw_images \
  --save_dir /home/cvlab/Desktop/2025/quantization/outputs \
  --tar_name cloud_results \
  --name new_L71025040_04020010723_6.png old_L71025040_04020010723_3.png

"""

# =========================
# 로깅 설정
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/cloud_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# 전역 설정
# =========================
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249],
                std=[0.1129, 0.0990, 0.0898]),
])


# =========================
# 모델 로드
# =========================
def load_model(weights_path: str, num_classes: int = 2, device: torch.device = DEVICE):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval().to(device)
    return model


# =========================
# 타일 좌표 생성
# =========================
def make_tile_coords(W: int, H: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    stride = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if xs[-1] != W - tile:
        xs.append(W - tile)
    if ys[-1] != H - tile:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


# =========================
# 블렌딩 창
# =========================
def make_blend_window(tile: int, overlap: int) -> np.ndarray:
    if overlap == 0:
        return np.ones((tile, tile), dtype=np.float32)
    w_1d = np.hanning(tile)
    w_1d = np.clip(w_1d, 1e-2, None)
    return np.outer(w_1d, w_1d).astype(np.float32)


# =========================
# 추론 (타일 기반)
# =========================
@torch.inference_mode()
def infer_tiled_image(
    image_path: str,
    model,
    save_dir: str,
    out_basename: str = "mask",
    num_classes: int = 2,
    tile_size: int = 384,
    overlap: int = 32,
    batch_size: int = 4,
    use_amp: bool = True,
    out_ext: str = ".png",
):
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    coords = make_tile_coords(W, H, tile_size, overlap)
    blend_win = make_blend_window(tile_size, overlap)

    acc = np.zeros((num_classes, H, W), dtype=np.float32)
    acc_w = np.zeros((H, W), dtype=np.float32)
    tile_tensors, tile_positions = [], []

    amp_dtype = torch.float16 if (use_amp and DEVICE.type == "cuda") else None

    def flush_batch():
        nonlocal tile_tensors, tile_positions
        if not tile_tensors:
            return
        x = torch.stack(tile_tensors, dim=0).to(DEVICE)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)["out"]
        else:
            logits = model(x)["out"]
        logits = logits.detach().to(torch.float32).cpu().numpy()
        for i, (left, top) in enumerate(tile_positions):
            acc[:, top:top+tile_size, left:left+tile_size] += logits[i] * blend_win[None, :, :]
            acc_w[top:top+tile_size, left:left+tile_size] += blend_win
        tile_tensors, tile_positions = [], []

    for (left, top) in coords:
        t = preprocess(img.crop((left, top, left+tile_size, top+tile_size)))
        tile_tensors.append(t)
        tile_positions.append((left, top))
        if len(tile_tensors) >= batch_size:
            flush_batch()
    flush_batch()

    acc /= np.clip(acc_w[None, :, :], 1e-6, None)
    mask = (np.argmax(acc, axis=0).astype(np.uint8) * 255)

    out_path = os.path.join(save_dir, f"{out_basename}{out_ext}")
    Image.fromarray(mask, mode="L").save(out_path)
    logger.info(f"[Saved mask] {out_path}")
    return out_path


# =========================
# 추론 + 패키징
# =========================
def run_inference_and_package(model, image_path, save_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_dir = os.path.join(save_dir, image_name)
    os.makedirs(img_save_dir, exist_ok=True)

    # 추론 시간 측정
    start = time.time()
    mask_path = infer_tiled_image(
        image_path=image_path,
        model=model,
        save_dir=img_save_dir,
        out_basename="mask",
        num_classes=2,
        tile_size=384,
        overlap=32,
        batch_size=4,
        use_amp=True,
        out_ext=".png",
    )
    end = time.time()
    inference_time = (end - start) * 1000  # ms

    # 원본 저장
    orig_path = os.path.join(img_save_dir, "original.png")
    shutil.copy(image_path, orig_path)

    # UTC 시간
    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # 구름 비율 계산
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cloud_ratio = (mask == 255).sum() / mask.size * 100

    # metadata.txt 저장
    meta_path = os.path.join(img_save_dir, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"Image Name: {image_name}\n")
        f.write(f"Inference Time: {inference_time:.2f} ms\n")
        f.write(f"UTC Time: {utc_now}\n")
        f.write(f"Cloud Coverage: {cloud_ratio:.2f}%\n")

    return img_save_dir


# =========================
# 전체 실행 (tar 묶기)
# =========================
def package_all(model, image_paths, save_dir, tar_name="cloud_results"):
    all_dirs = []
    for img_path in image_paths:
        print(" -", img_path)
        result_dir = run_inference_and_package(model, img_path, save_dir)
        all_dirs.append(result_dir)

    # hd_bin.txt (루트에 1개만 생성)
    hd_bin_path = os.path.join(save_dir, "hd_bin.txt")
    open(hd_bin_path, "w").close()

    # tar 묶기
    tar_path = f"{tar_name}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for d in all_dirs:
            tar.add(d, arcname=os.path.basename(d))
        tar.add(hd_bin_path, arcname="hd_bin.txt")

    print(f"[OK] Packaging completed: {tar_path}")
    return tar_path


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Detection Inference & Packaging")

    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to model weights (.pth)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory where input images are located")
    parser.add_argument("--save_dir", type=str, default="./quantization/outputs",
                        help="Directory to save outputs")
    parser.add_argument("--tar_name", type=str, default="cloud_results",
                        help="Name of output tar file (without extension)")
    parser.add_argument("--name", type=str, nargs="+", required=True,
                        help="One or more image names (inside image_dir)")

    args = parser.parse_args()

    # 모델 로드
    model = load_model(args.weights_path, num_classes=2, device=DEVICE)

    # 이미지 경로 구성
    image_paths = [os.path.join(args.image_dir, n) for n in args.name]

    # 패키징 실행
    package_all(model, image_paths, args.save_dir, tar_name=args.tar_name)