import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


# -------------------------------------------------------
# 보드별 설정 (Xavier/Orin을 기본 활성화, Nano는 주석 처리)
# -------------------------------------------------------
# --- Jetson Nano 권장 설정 ---
# BOARD = "NANO"
# BATCH_SIZE = 2
# USE_AMP = False          # Nano(Maxwell)는 AMP로 속도 이득이 적음(메모리 절감 목적으론 True 가능)
# USE_CHANNELS_LAST = False

# --- Jetson Xavier/Orin 권장 설정 --- (기본값)
BOARD = "XAVIER_ORIN"
BATCH_SIZE = 8            # Orin: 8~16, Xavier: 8 전후
USE_AMP = True
USE_CHANNELS_LAST = True


# -------------------------------------------------------
# 모델 로드
# -------------------------------------------------------
def load_model(weights_path: str, num_classes: int, device: torch.device):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return model


# -------------------------------------------------------
# 인메모리 패딩: (H,W,3) → (Hpad,Wpad,3), 좌상단에 원본 복사
# -------------------------------------------------------
def pad_image_numpy(np_img: np.ndarray, tile_size: int):
    H, W, C = np_img.shape
    Wpad = math.ceil(W / tile_size) * tile_size
    Hpad = math.ceil(H / tile_size) * tile_size
    if (Hpad == H) and (Wpad == W):
        return np_img, (H, W), (Hpad, Wpad)

    canvas = np.zeros((Hpad, Wpad, C), dtype=np.uint8)
    canvas[:H, :W] = np_img
    return canvas, (H, W), (Hpad, Wpad)


# -------------------------------------------------------
# 타일 배치 제너레이터: 고정 크기 타일만 생성(경계는 패딩으로 보장)
# -------------------------------------------------------
def tile_batch_generator_fixed(np_img_pad: np.ndarray, tile_size: int, batch_size: int):
    Hpad, Wpad, _ = np_img_pad.shape
    nx = Wpad // tile_size
    ny = Hpad // tile_size
    batch, coords = [], []

    for ty in range(ny):
        y0 = ty * tile_size
        for tx in range(nx):
            x0 = tx * tile_size
            patch = np_img_pad[y0:y0 + tile_size, x0:x0 + tile_size, :]
            batch.append(patch)
            coords.append((x0, y0))
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0), coords
                batch, coords = [], []
    if batch:
        yield np.stack(batch, axis=0), coords


# -------------------------------------------------------
# 추론(+ AMP + GPU 정규화) → 전체 마스크 조립 → 원본 크기 크롭 저장
# -------------------------------------------------------
@torch.inference_mode()
def infer_tiled_and_save_mask(image_path: str,
                              model,
                              device: torch.device,
                              save_dir: str,
                              tile_size: int = 384,
                              batch_size: int = 8,
                              use_amp: bool = True,
                              use_channels_last: bool = True):
    os.makedirs(save_dir, exist_ok=True)

    # 1) 입력 로드 & 인메모리 패딩
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img, dtype=np.uint8)           # (H,W,3)
    np_img_pad, (H, W), (Hpad, Wpad) = pad_image_numpy(np_img, tile_size)

    # 2) 출력 버퍼(패딩 크기 기준) 준비
    full_mask_pad = np.zeros((Hpad, Wpad), dtype=np.uint8)

    # 3) 정규화 파라미터 (GPU에서 브로드캐스트)
    mean = torch.tensor([0.4943, 0.4672, 0.5249], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.1129, 0.0990, 0.0898], device=device).view(1, 3, 1, 1)

    # 4) 배치 추론 루프 (모든 타일은 고정 크기)
    for np_batch, coords in tile_batch_generator_fixed(np_img_pad, tile_size, batch_size):
        x = torch.from_numpy(np_batch).to(device, non_blocking=True)  # (B,ts,ts,3)
        x = x.permute(0, 3, 1, 2)                                     # (B,3,ts,ts)
        if use_channels_last:
            x = x.contiguous().to(memory_format=torch.channels_last)

        x = x.float().div_(255.0)
        x = (x - mean) / std

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda' and use_amp)):
            logits = model(x)["out"]                                  # (B,C,ts,ts)
        pred = logits.argmax(dim=1)                                   # (B,ts,ts), int64
        pred_np = (pred.to('cpu', non_blocking=True).numpy().astype(np.uint8)) * 255

        # 5) 패딩 버퍼에 그대로 배치 삽입(모두 고정 크기)
        for i, (x0, y0) in enumerate(coords):
            full_mask_pad[y0:y0 + tile_size, x0:x0 + tile_size] = pred_np[i]

    # 6) 원본 크기(H,W)로 최종 크롭 후 저장
    full_mask = full_mask_pad[:H, :W]
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_mask_{W}x{H}.tif")
    Image.fromarray(full_mask, mode='L').save(out_path, format="TIFF", compression="tiff_lzw")
    print(f"[Saved] {out_path}")
    return out_path


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    image_path = "L71025040_04020010723.TIF"
    weights_path = "outputs/weights/best-mobilenetv3-restylized_irish_400-finetune_head+backbone_1e5-v1.pth"
    save_dir = "cloud_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, num_classes=2, device=device)

    # 보드별 설정 적용 (기본: Xavier/Orin, 필요 시 위의 Nano 블록 활성화)
    infer_tiled_and_save_mask(
        image_path=image_path,
        model=model,
        device=device,
        save_dir=save_dir,
        tile_size=384,
        batch_size=BATCH_SIZE,
        use_amp=USE_AMP,
        use_channels_last=USE_CHANNELS_LAST
    )
