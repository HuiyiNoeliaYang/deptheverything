# test_case.py — metric depth → 16-bit PNG (factor=1000, 1 unit = 1 mm)
import os, sys, cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from depth_anything_v2.dpt import DepthAnythingV2

DATASET = 'hypersim'                # 'hypersim' (indoor, 20 m) | 'vkitti' (outdoor, 80 m)
MAX_DEPTH_BY_DATASET = {'hypersim': 20, 'vkitti': 80}

ENCODER, FACTOR, INPUT_SIZE = 'vitl', 1000, 518
MAX_DEPTH = MAX_DEPTH_BY_DATASET[DATASET]
CKPT = os.path.join(HERE, 'checkpoints',
                    f'depth_anything_v2_metric_{DATASET}_{ENCODER}.pth')

CFG = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
DEVICE = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available() else 'cpu')

model = DepthAnythingV2(**{**CFG[ENCODER], 'max_depth': MAX_DEPTH})
model.load_state_dict(torch.load(CKPT, map_location='cpu'))
model = model.to(DEVICE).eval()

def make_depth_png(rgb_path: str, out_path: str):
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise FileNotFoundError(f'cv2.imread returned None for {rgb_path!r} '
                                f'(does the file exist? is it a readable image?)')
    depth_m = model.infer_image(bgr, INPUT_SIZE)        # float32 meters
    valid = (depth_m > 0) & (depth_m < MAX_DEPTH - 1e-3)
    out = np.zeros_like(depth_m, dtype=np.uint16)
    out[valid] = np.clip(depth_m[valid] * FACTOR, 0, 65535).astype(np.uint16)
    cv2.imwrite(out_path, out)
    print(f'{rgb_path} -> {out_path}  shape={out.shape} dtype={out.dtype} '
          f'min={out.min()} max={out.max()}')

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

def iter_images(in_path: str):
    if os.path.isfile(in_path):
        yield in_path, os.path.basename(in_path)
        return
    if not os.path.isdir(in_path):
        raise FileNotFoundError(f'{in_path!r} is neither a file nor a directory')
    for root, _, files in os.walk(in_path):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, in_path)
                yield full, rel

def run(in_path: str, out_path: str):
    items = list(iter_images(in_path))
    if not items:
        raise FileNotFoundError(f'no images found under {in_path!r}')

    treat_out_as_dir = os.path.isdir(in_path) or len(items) > 1
    if treat_out_as_dir:
        os.makedirs(out_path, exist_ok=True)

    for i, (src, rel) in enumerate(items, 1):
        if treat_out_as_dir:
            dst = os.path.join(out_path, os.path.splitext(rel)[0] + '.png')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        else:
            stem, ext = os.path.splitext(out_path)
            dst = out_path if ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp') else stem + '.png'
        print(f'[{i}/{len(items)}]', end=' ')
        make_depth_png(src, dst)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python test_case.py <image_or_dir> [output_path]')
        sys.exit(1)
    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else (
        'depth_out' if os.path.isdir(in_path) else 'depth.png')
    run(in_path, out_path)

