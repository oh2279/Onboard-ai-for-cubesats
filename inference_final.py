import os
import argparse
from typing import List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


# 전역 설정
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 전처리
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
])


# 모델 로드
def load_model(weights_path: str, num_classes: int, device: torch.device = DEVICE):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval().to(device)
    return model


# 타일 좌표 생성
def make_tile_coords(W: int, H: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    stride = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if xs[-1] != W - tile:
        xs.append(W - tile)
    if ys[-1] != H - tile:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


# 블렌딩 창
def make_blend_window(tile: int, overlap: int) -> np.ndarray:
    if overlap == 0:
        return np.ones((tile, tile), dtype=np.float32)
    w_1d = np.hanning(tile)
    w_1d = np.clip(w_1d, 1e-2, None)
    return np.outer(w_1d, w_1d).astype(np.float32)


# 추론
@torch.inference_mode()
def infer_tiled_image(
    image_path: str,
    model,
    save_dir: str,
    out_basename: str,
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
    print(f"[Saved mask] {out_path}")
    return out_path


# 파일명 처리
def new_to_old_name(path: Path) -> Path:
    if path.name.startswith("new_"):
        return path.with_name("old_" + path.name[len("new_"):])
    return path

def strip_prefix(name: str) -> str:
    if name.startswith("new_"):
        return name[len("new_"):]
    if name.startswith("old_"):
        return name[len("old_"):]
    return name


# dir 처리
def process_directory(input_dir: str, save_dir: str, model, **kwargs):
    input_dir, save_dir = Path(input_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(input_dir.glob("new_*.png"))
    if not targets:
        print("[INFO] no new_*.png found.")
        return

    for src in targets:
        try:
            dst = new_to_old_name(src)
            clean_name = strip_prefix(dst.stem)
            infer_tiled_image(
                image_path=str(src),
                model=model,
                save_dir=str(save_dir),
                out_basename=clean_name,
                **kwargs
            )
            if not dst.exists():
                src.rename(dst)
                print(f"[RENAMED] {src.name} -> {dst.name}")
            else:
                print(f"[WARN] target exists, skip rename: {dst.name}")
        except Exception as e:
            print(f"[ERROR] {src.name}: {e}")


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile-based image inference")
    parser.add_argument("--input_dir", type=str, default="raw_images", help="Input image directory")
    parser.add_argument("--save_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--weights_path", type=str, default="current/best-mobilenetv3-restylized_irish_400-finetune_head+backbone_1e5-v1.pth", help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")

    args = parser.parse_args()

    model = load_model(args.weights_path, num_classes=2, device=DEVICE)

    process_directory(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
        model=model,
        num_classes=2,
        tile_size=384,
        overlap=32,
        batch_size=args.batch_size,
        use_amp=True,
    )
