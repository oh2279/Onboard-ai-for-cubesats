from PIL import Image
import os
import random

# 입력 이미지 경로
input_path = "/home/jovyan/cvlab/MobileNetV3/L71025040_04020010723.TIF"
# 출력 저장 폴더
output_dir = "/home/jovyan/cvlab/MobileNetV3/random_patches"
os.makedirs(output_dir, exist_ok=True)

# 자를 크기
patch_w, patch_h = 2464, 2056

# 열고 크기 확인
img = Image.open(input_path)
W, H = img.size
print("원본 크기:", W, H)

# 랜덤 패치 10개 생성
num_patches = 10
for i in range(num_patches):
    # 랜덤 좌표 (우측/하단 넘어가지 않도록 제한)
    left = random.randint(0, W - patch_w)
    upper = random.randint(0, H - patch_h)
    right = left + patch_w
    lower = upper + patch_h

    crop = img.crop((left, upper, right, lower))
    out_path = os.path.join(output_dir, f"patch_{i+1}.png")
    crop.save(out_path, "PNG")
    print(f"저장 완료: {out_path}")

print(f"총 {num_patches}개의 랜덤 패치 생성 완료.")
