import os
from PIL import Image

def split_tiff_to_tiles(image_path: str, save_dir: str, tile_size: int = 384):
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".tif", ".tiff"):
        raise ValueError("TIFF 파일만 지원합니다.")

    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    if W % tile_size != 0 or H % tile_size != 0:
        raise ValueError(f"이미지 크기 ({W}x{H})가 {tile_size} 배수가 아닙니다. 먼저 패딩하세요.")

    tiles_x, tiles_y = W // tile_size, H // tile_size
    base = os.path.splitext(os.path.basename(image_path))[0]

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            left, top = tx * tile_size, ty * tile_size
            right, bottom = left + tile_size, top + tile_size
            tile = img.crop((left, top, right, bottom))

            tile_name = f"{base}_tile_{ty}_{tx}.tif"
            tile_path = os.path.join(save_dir, tile_name)
            tile.save(tile_path, format="tiff")
            print(f"[Saved] {tile_path}")

if __name__ == "__main__":
    image_path = "padded_1152x1152.tif"
    save_dir = "tiles_384x384"
    split_tiff_to_tiles(image_path, save_dir)
