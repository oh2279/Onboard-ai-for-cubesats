import os
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


# 전처리 (학습과 동일)
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
])


def load_model(weights_path: str, num_classes: int, device: torch.device):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def infer_tiled_tiff(image_path: str, model, device: torch.device, save_dir: str,
                     tile_size: int = 384):
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".tif", ".tiff"):
        raise ValueError("TIFF 파일만 지원합니다.")
    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    if (W % tile_size != 0) or (H % tile_size != 0):
        raise ValueError(f"이미지 크기({W}x{H})가 {tile_size}의 배수가 아닙니다. 먼저 패딩하세요.")

    tiles_x, tiles_y = W // tile_size, H // tile_size
    full_mask = Image.new("L", (W, H), 0)  # 0/255 이진 마스크

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            left, top = tx * tile_size, ty * tile_size
            tile = img.crop((left, top, left + tile_size, top + tile_size))

            x = preprocess(tile).unsqueeze(0).to(device)      # (1,3,384,384)
            logits = model(x)["out"]                          # (1,C,384,384)
            pred = logits.argmax(dim=1).byte().squeeze(0)     # (384,384), {0,1}
            tile_mask = to_pil_image(pred * 255)              # 'L' 0/255

            full_mask.paste(tile_mask, (left, top))

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"{base}_mask_{W}x{H}.tif")
    full_mask.save(out_path, format="TIFF")
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    image_path = "padded_1152x1152.tif"  # 384의 배수 크기(예: 1152x1152)
    weights_path = "outputs/weights/best-mobilenetv3-restylized_irish_400-own_finetune_1e3-v1.pth"
    save_dir = "cloud_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, num_classes=2, device=device)
    infer_tiled_tiff(image_path, model, device, save_dir)
