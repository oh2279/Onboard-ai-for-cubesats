import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


# ------------------------------
# Preprocess
# ------------------------------
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
])


# ------------------------------
# Model Loader
# ------------------------------
def load_model(weights_path, num_classes, device):
    model = lraspp_mobilenet_v3_large(pretrained=False)
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1)
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ------------------------------
# Single-image Inference
# ------------------------------
def infer_single_image(image_path, model, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1) 원본 로드 & 크기 확보
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    # 2) 전처리 → (1,3,384,384)
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    # 3) 추론
    with torch.no_grad():
        outputs = model(input_tensor)          # dict with 'out'
        logits = outputs["out"]                # (1,C,384,384)
        pred = torch.argmax(logits, dim=1)     # (1,384,384)

    # 4) 원본 크기로 업샘플 (nearest)
    pred_up = F.interpolate(
        pred.float().unsqueeze(1),  # (1,1,h,w)
        size=(orig_h, orig_w),
        mode="nearest"
    ).squeeze(1).squeeze(0).byte()  # (H,W), {0,1}

    # 5) 이진 마스크 저장 (0->0, 1->255)
    mask_viz = (pred_up * 255)
    mask_pil = to_pil_image(mask_viz)
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(save_dir, f"{base}_mask.png")
    mask_pil.save(mask_path)

    # 6) 오버레이 생성 (빨강, 알파=128)
    overlay = Image.new("RGBA", (orig_w, orig_h), (255, 0, 0, 0))
    alpha_pil = to_pil_image(pred_up * 128)        # 0 or 128의 'L' 알파
    overlay.putalpha(alpha_pil)
    overlayed = Image.alpha_composite(image_pil.convert("RGBA"), overlay)

    overlay_path = os.path.join(save_dir, f"{base}_overlay.png")
    overlayed.save(overlay_path)

    print(f"[Saved] mask:    {mask_path}")
    print(f"[Saved] overlay: {overlay_path}")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    image_path = "/home/jovyan/cvlab/MobileNetV3/L71025040_04020010723.TIF"
    weights_path = "outputs/weights/best-mobilenetv3-restylized_irish_400-own_finetune_1e3-v1.pth"
    save_dir = "cloud_outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    model = load_model(weights_path, num_classes, device)
    infer_single_image(image_path, model, device, save_dir)
