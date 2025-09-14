import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

# ===== 설정 =====
num_classes = 2
patch_size = 384
overlap = 64  # 0~128 사이 추천. 0이면 비오버랩(경계 사각형 가능성↑)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 입력 크기 고정 → 약간 유리

transform_image = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4943, 0.4672, 0.5249], std=[0.1129, 0.0990, 0.0898]),
])

# ===== 모델 =====
model = lraspp_mobilenet_v3_large(pretrained=False)
model.classifier.low_classifier  = nn.Conv2d(40,  num_classes, kernel_size=1)
model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1)
model.to(device).eval()
state = torch.load("outputs/weights/best-mobilenetv3-restylized_irish_400-own_finetune_1e3-v1.pth",
                   map_location=device)
model.load_state_dict(state)

# ===== 이미지 =====
img = Image.open("L71025040_04020010723_2_1.png").convert("RGB")
W, H = img.size  # 4006 x 3595 (W x H)

# ===== 오버랩 그리드 만들기 (항상 384x384만 자르도록) =====
def make_grid(L, win, ov):
    stride = win - ov
    xs = list(range(0, max(L - win, 0) + 1, stride))
    if xs[-1] != L - win:
        xs.append(L - win)
    return xs

xs = make_grid(W, patch_size, overlap)
ys = make_grid(H, patch_size, overlap)

# ===== 블렌딩 가중치(2D) =====
if overlap > 0:
    w1d = np.hanning(patch_size).astype(np.float32)
    w1d = np.clip(w1d, 0.05, None)  # 0 가중치 방지(가장자리 단독 영역도 커버)
else:
    w1d = np.ones(patch_size, dtype=np.float32)
w2d = np.outer(w1d, w1d).astype(np.float32)

# ===== 누적 버퍼(로짓 블렌딩) =====
accum_logits = np.zeros((num_classes, H, W), dtype=np.float32)
accum_weight = np.zeros((H, W), dtype=np.float32)

# ===== 추론 & 합치기 =====
with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
    for y in ys:
        for x in xs:
            patch = img.crop((x, y, x + patch_size, y + patch_size))  # 항상 384x384
            inp = transform_image(patch).unsqueeze(0).to(device)
            out = model(inp)['out'][0].detach().cpu().numpy()  # (C, 384, 384) 로짓

            # 로짓 가중 합
            accum_logits[:, y:y+patch_size, x:x+patch_size] += out * w2d
            accum_weight[y:y+patch_size, x:x+patch_size] += w2d

# 가중치 나눗셈 (안전 분모)
eps = 1e-6
accum_logits /= (accum_weight[None, :, :] + eps)

# 최종 예측
pred = np.argmax(accum_logits, axis=0).astype(np.uint8)

# 저장
Image.fromarray((pred * 255).astype(np.uint8)).save("segmentation_result_reconstructed.png")
