import os
import math
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


# ------------------------------
# 전처리
# ------------------------------
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------
# 모델 로드
# ------------------------------
def load_model(weights_path: str, num_classes: int, device: torch.device):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ------------------------------
# (1) TIFF 패딩
# ------------------------------
def pad_tiff(image_path: str, save_path: str, block_size=384):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    new_w = math.ceil(orig_w / block_size) * block_size
    new_h = math.ceil(orig_h / block_size) * block_size

    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))  # 검은색 패딩
    padded.paste(img, (0, 0))
    padded.save(save_path, format="TIFF")

    print(f"[Saved padded TIFF] {save_path} ({orig_w}x{orig_h} → {new_w}x{new_h})")
    return save_path, (orig_w, orig_h), (new_w, new_h)


# ------------------------------
# (2) 타일 추론 후 (3) 최종 마스크 원본 크기로 저장
# ------------------------------
@torch.no_grad()
def infer_tiled_tiff(image_path: str, model, device: torch.device, save_dir: str,
                     orig_size, tile_size: int = 384):
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    tiles_x, tiles_y = W // tile_size, H // tile_size
    full_mask = Image.new("L", (W, H), 0)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            left, top = tx * tile_size, ty * tile_size
            tile = img.crop((left, top, left + tile_size, top + tile_size))

            x = preprocess(tile).unsqueeze(0).to(device)
            logits = model(x)["out"]
            pred = logits.argmax(dim=1).byte().squeeze(0)    # (384,384), {0,1}
            tile_mask = to_pil_image(pred * 255)

            full_mask.paste(tile_mask, (left, top))

    # 원본 크기로 크롭 (패딩 부분 제거)
    orig_w, orig_h = orig_size
    full_mask_cropped = full_mask.crop((0, 0, orig_w, orig_h))

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_mask_{orig_w}x{orig_h}.tif")
    full_mask_cropped.save(out_path, format="TIFF")
    print(f"[Saved full mask (cropped to original size)] {out_path}")
    return out_path


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    image_path = "L71025040_04020010723.TIF"
    weights_path = "outputs/weights/best-mobilenetv3-restylized_irish_400-finetune_head+backbone_1e5-v1.pth"
    save_dir = "cloud_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, num_classes=2, device=device)

    # (1) TIFF 패딩
    padded_path, orig_size, padded_size = pad_tiff(
        image_path, os.path.join(save_dir, "padded.tif")
    )

    # (2)+(3) 타일 추론 후 최종 마스크는 원본 크기로 저장
    infer_tiled_tiff(padded_path, model, device, save_dir, orig_size)
