from PIL import Image
import os

# Import the richer runner (with EXIF) from the Gradio wrapper
from app_gradio import run_process_with_exif

# Create a small synthetic test image (RGB gradient)
img = Image.new('RGB', (128, 128))
for y in range(128):
    for x in range(128):
        img.putpixel((x, y), (int(x*2), int(y*2), int((x+y)/2)))

# Call with explicit parameters to exercise many code paths
out_img, status, exif = run_process_with_exif(
    img,
    noise_std=0.02,
    clahe_clip=2.0,
    tile=8,
    cutoff=0.25,
    fstrength=0.9,
    awb=True,
    sim_camera=True,
    lut_file=None,
    lut_strength=0.1,
    awb_ref=None,
    fft_ref=None,
    seed=0,
    jpeg_cycles=1,
    jpeg_qmin=88,
    jpeg_qmax=96,
    vignette_strength=0.35,
    chroma_strength=1.2,
    iso_scale=1.0,
    read_noise=2.0,
    no_no_bayer=False,
    randomness=0.05,
    perturb=0.008,
    phase_perturb=0.08,
    radial_smooth=5,
    fft_mode="auto",
    fft_alpha=1.0,
    hot_pixel_prob=1e-6,
    banding_strength=0.0,
    motion_blur_kernel=1,
)

print('Status:', status)
print('EXIF (hex length):', len(exif) if exif else 0)
if out_img is not None:
    os.makedirs('test', exist_ok=True)
    out_path = os.path.join('test', 'test_gradio_output_debug.jpg')
    out_img.save(out_path)
    print('Wrote output to', out_path)
else:
    print('No output image generated')
