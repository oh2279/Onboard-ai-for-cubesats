import os
import math
from PIL import Image

def pad_to_multiple_of_384(image_path: str, save_path: str, block_size=384):
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".tif", ".tiff"):
        raise ValueError("TIFF 파일만 지원합니다.")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 384 배수로 올림
    new_w = math.ceil(w / block_size) * block_size
    new_h = math.ceil(h / block_size) * block_size

    # 검은색 배경 생성 후 원본 복사
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(img, (0, 0))

    padded.save(save_path, format="tiff")

    print(f"[Saved] {save_path}")
    print(f"원본 크기: ({w}, {h}) → 패딩 후 크기: ({new_w}, {new_h})")


if __name__ == "__main__":
    image_path = "random_crop_1002x898.tif"
    save_path = "1152x1152.tif"
    pad_to_multiple_of_384(image_path, save_path)
