from PIL import Image
import numpy as np
from app_gradio import run_process_with_exif


def mse(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    return float(((a - b) ** 2).mean())


def make_input():
    img = Image.new('RGB', (256, 256))
    for y in range(256):
        for x in range(256):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    return img


def run_variant(name, **kwargs):
    inp = make_input()
    out, status, exif = run_process_with_exif(inp, **kwargs)
    import os
    os.makedirs('test', exist_ok=True)
    path = os.path.join('test', f"test_out_{name}.jpg")
    if out is not None:
        out.save(path)
    print(f"{name}: status={status}, exif_len={len(exif) if exif else 0}, mse={mse(inp, out) if out is not None else 'NA'}, saved={path if out is not None else 'no'}")


if __name__ == '__main__':
    # Default
    run_variant('default', noise_std=0.02, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9, awb=True, sim_camera=True, seed=0)
    # AWB disabled
    run_variant('no_awb', noise_std=0.02, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9, awb=False, sim_camera=True, seed=0)
    # camera sim disabled
    run_variant('no_cam', noise_std=0.02, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9, awb=True, sim_camera=False, seed=0)
    # both disabled
    run_variant('no_awb_no_cam', noise_std=0.02, clahe_clip=2.0, tile=8, cutoff=0.25, fstrength=0.9, awb=False, sim_camera=False, seed=0)
