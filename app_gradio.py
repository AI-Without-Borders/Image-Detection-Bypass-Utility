"""
Gradio web UI wrapper for the Image Postprocess pipeline.

This lightweight wrapper saves the uploaded image to a temporary file,
constructs a minimal args namespace expected by `process_image`, runs the
processing pipeline, and returns the result to the browser.

Designed for quick deployment on Huggingface Spaces.
"""
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace
from typing import Optional
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import json

# Preset persistence file (in repo root)
PRESETS_FILE = Path(__file__).parent / "presets.json"

# Builtin presets
BUILTIN_PRESETS = {
    "Default": {},
    "NovaNodes (reference)": {
    "noise_std": 0.02,
    "clahe_clip": 2.0,
    "tile": 8,
    "cutoff": 0.25,
    "fstrength": 0.9,
    "randomness": 0.05,
    # align perturb with NovaNodes default
    "perturb": 0.01,
    "phase_perturb": 0.08,
    "radial_smooth": 5,
    "jpeg_cycles": 1,
    "jpeg_qmin": 88,
    # camera simulation enabled by default for 'reference'
    "sim_camera": True,
    "vignette_strength": 0.35,
    "chroma_strength": 1.2,
    "iso_scale": 1.0,
    "read_noise": 2.0,
    # align hot pixel probability with NovaNodes default
    "hot_pixel_prob": 1e-7,
    "no_no_bayer": False,
    # align LUT strength to node default (1.0)
    "lut_strength": 1.0,
    },
    "High JPEG cycles": {"jpeg_cycles": 3, "jpeg_qmin": 70},
    "Aggressive": {"noise_std": 0.06, "fstrength": 1.0, "perturb": 0.02, "jpeg_cycles": 2},
    "Subtle": {"noise_std": 0.01, "fstrength": 0.6},

    # Conservative preview preset (closer to input image): disables camera simulation
    "Preview (no camera sim)": {"sim_camera": False, "noise_std": 0.01, "fstrength": 0.6},
}


def load_custom_presets():
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_custom_preset(name: str, data: dict):
    presets = load_custom_presets()
    presets[name] = data
    PRESETS_FILE.write_text(json.dumps(presets, indent=2))


def get_preset_overrides(name: str):
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name].copy()
    customs = load_custom_presets()
    return customs.get(name, {}).copy()


def preset_summary(name: str):
    overrides = get_preset_overrides(name)
    if not overrides:
        return "(no overrides)"
    return json.dumps(overrides, indent=2)

gr = None

try:
    from image_postprocess import process_image
except Exception as e:
    process_image = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None


def _mk_temp_file(suffix: str = ".png") -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return f.name


def run_process(
    img: Image.Image,
    noise_std: float = 0.02,
    clahe_clip: float = 2.0,
    tile: int = 8,
    cutoff: float = 0.25,
    fstrength: float = 0.9,
    awb: bool = True,
    sim_camera: bool = True,
    lut_file: Optional[Path] = None,
    lut_strength: float = 0.1,
):
    """Run the repository's processing pipeline on a PIL image and return a PIL image.

    Returns (pil.Image or None, status string).
    """
    if process_image is None:
        return None, f"Backend import error: {IMPORT_ERROR}"

    tmp_files = []
    try:
        in_path = _mk_temp_file(suffix=".png")
        img.save(in_path)
        tmp_files.append(in_path)

        out_path = _mk_temp_file(suffix=".jpg")
        tmp_files.append(out_path)

        lut_path = None
        if lut_file is not None:
            # gr.File gives a pathlib.Path-like object; accept either str or Path
            lut_path = str(lut_file)

        args = SimpleNamespace(
            input=in_path,
            output=out_path,
            awb=bool(awb),
            ref=None,
            noise_std=float(noise_std),
            clahe_clip=float(clahe_clip),
            tile=int(tile),
            cutoff=float(cutoff),
            fstrength=float(fstrength),
            randomness=0.05,
            perturb=0.008,
            seed=None,
            fft_ref=None,
            fft_mode="auto",
            fft_alpha=1.0,
            phase_perturb=0.08,
            radial_smooth=5,
            sim_camera=bool(sim_camera),
            no_no_bayer=False,
            jpeg_cycles=1,
            jpeg_qmin=88,
            jpeg_qmax=96,
            vignette_strength=0.35,
            chroma_strength=1.2,
            iso_scale=1.0,
            read_noise=2.0,
            hot_pixel_prob=1e-6,
            banding_strength=0.0,
            motion_blur_kernel=1,
            lut=(lut_path if lut_path else None),
            lut_strength=float(lut_strength),
        )

        # Debug: print args passed to process_image for tracing
        try:
            print("process_image called with args:")
            for k, v in vars(args).items():
                print(f"  {k}: {v}")
        except Exception:
            pass

        try:
            process_image(in_path, out_path, args)
        except Exception as e:
            return None, f"Processing error: {e}"

        out_img = Image.open(out_path).convert("RGB")
        return out_img, "OK"

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass


def run_process_with_exif(
    img: Image.Image,
    noise_std: float = 0.02,
    clahe_clip: float = 2.0,
    tile: int = 8,
    cutoff: float = 0.25,
    fstrength: float = 0.9,
    awb: bool = True,
    sim_camera: bool = True,
    lut_file: Optional[Path] = None,
    lut_strength: float = 0.1,
    awb_ref: Optional[Image.Image] = None,
    fft_ref: Optional[Image.Image] = None,
    seed: Optional[int] = None,
    jpeg_cycles: int = 1,
    jpeg_qmin: int = 88,
    jpeg_qmax: int = 96,
    vignette_strength: float = 0.35,
    chroma_strength: float = 1.2,
    iso_scale: float = 1.0,
    read_noise: float = 2.0,
    no_no_bayer: bool = False,
    randomness: float = 0.05,
    perturb: float = 0.008,
    phase_perturb: float = 0.08,
    radial_smooth: int = 5,
    fft_mode: str = "auto",
    fft_alpha: float = 1.0,
    apply_exif: bool = True,
    hot_pixel_prob: float = 1e-6,
    banding_strength: float = 0.0,
    motion_blur_kernel: int = 1,
):
    """Run pipeline like `run_process` but return (pil_img, status, exif_hex_or_empty).

    This function is used by the Gradio UI to expose EXIF metadata.
    """
    if process_image is None:
        return None, f"Backend import error: {IMPORT_ERROR}", ""

    tmp_files = []
    try:
        in_path = _mk_temp_file(suffix=".png")
        img.save(in_path)
        tmp_files.append(in_path)

        # optional refs
        awb_ref_path = None
        if awb_ref is not None:
            p = _mk_temp_file(suffix=".png")
            awb_ref.save(p)
            awb_ref_path = p
            tmp_files.append(p)

        fft_ref_path = None
        if fft_ref is not None:
            p = _mk_temp_file(suffix=".png")
            fft_ref.save(p)
            fft_ref_path = p
            tmp_files.append(p)

        out_path = _mk_temp_file(suffix=".jpg")
        tmp_files.append(out_path)

        lut_path = None
        if lut_file is not None:
            lut_path = str(lut_file)

        args = SimpleNamespace(
            input=in_path,
            output=out_path,
            awb=bool(awb),
            ref=awb_ref_path,
            noise_std=float(noise_std),
            clahe_clip=float(clahe_clip),
            tile=int(tile),
            cutoff=float(cutoff),
            fstrength=float(fstrength),
            randomness=float(randomness),
            perturb=float(perturb),
            seed=seed,
            fft_ref=fft_ref_path,
            fft_mode=fft_mode,
            fft_alpha=float(fft_alpha),
            phase_perturb=float(phase_perturb),
            radial_smooth=int(radial_smooth),
            sim_camera=bool(sim_camera),
            no_no_bayer=bool(no_no_bayer),
            jpeg_cycles=int(jpeg_cycles),
            jpeg_qmin=int(jpeg_qmin),
            jpeg_qmax=int(jpeg_qmax),
            vignette_strength=float(vignette_strength),
            chroma_strength=float(chroma_strength),
            iso_scale=float(iso_scale),
            read_noise=float(read_noise),
            hot_pixel_prob=float(hot_pixel_prob),
            banding_strength=float(banding_strength),
            motion_blur_kernel=int(motion_blur_kernel),
            lut=(lut_path if lut_path else None),
            lut_strength=float(lut_strength),
        )

        # Debug: print args passed to process_image for tracing
        try:
            print("process_image called with args:")
            for k, v in vars(args).items():
                print(f"  {k}: {v}")
        except Exception:
            pass

        try:
            process_image(in_path, out_path, args)
        except Exception as e:
            return None, f"Processing error: {e}", ""

        out_img = Image.open(out_path).convert("RGB")

        # try to extract EXIF bytes
        exif_hex = ""
        try:
            info = Image.open(out_path).info
            exif_bytes = info.get('exif')
            if exif_bytes:
                exif_hex = exif_bytes.hex()
        except Exception:
            exif_hex = ""

        return out_img, "OK", exif_hex

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass

# ------------------ Headless analysis helpers (from AnalysisPanel) ------------------
def pil_to_gray_array(pil_img: Image.Image):
    arr = np.array(pil_img.convert('RGB'))
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    return gray


def compute_fft_magnitude(gray_arr, eps=1e-8):
    f = np.fft.fft2(gray_arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)
    return mag, mag_log


def radial_profile(mag, center=None, nbins=100):
    h, w = mag.shape
    if center is None:
        center = (int(h / 2), int(w / 2))
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_flat = r.ravel()
    mag_flat = mag.ravel()
    max_r = np.max(r_flat)
    if max_r <= 0:
        return np.linspace(0, 1, nbins), np.zeros(nbins)
    bins = np.linspace(0, max_r, nbins + 1)
    inds = np.digitize(r_flat, bins) - 1
    radial_mean = np.zeros(nbins)
    for i in range(nbins):
        sel = inds == i
        if np.any(sel):
            radial_mean[i] = mag_flat[sel].mean()
        else:
            radial_mean[i] = 0.0
    centers = 0.5 * (bins[:-1] + bins[1:]) / max_r
    return centers, radial_mean


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def make_analysis_images(pil_img: Image.Image):
    """Return (hist_img, fft_img, radial_img) as PIL Images for the provided PIL image."""
    gray = pil_to_gray_array(pil_img)

    # Histogram
    fig1 = plt.figure(figsize=(3, 2), dpi=100)
    ax1 = fig1.add_subplot(111)
    flat = gray.ravel()
    if flat.dtype.kind == 'f' and flat.max() <= 1.0:
        flat = (flat * 255.0).astype(np.uint8)
    ax1.hist(flat, bins=256, range=(0, 255))
    ax1.set_title('Grayscale histogram')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Count')
    hist_img = fig_to_pil(fig1)

    # FFT magnitude (log)
    mag, mag_log = compute_fft_magnitude(gray)
    fig2 = plt.figure(figsize=(3, 2), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(mag_log, origin='lower', aspect='auto')
    ax2.set_title('FFT magnitude (log)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fft_img = fig_to_pil(fig2)

    # Radial profile
    centers, radial = radial_profile(mag)
    fig3 = plt.figure(figsize=(3, 2), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.plot(centers, radial)
    ax3.set_title('Radial freq profile')
    ax3.set_xlabel('Normalized radius')
    ax3.set_ylabel('Mean magnitude')
    radial_img = fig_to_pil(fig3)

    return hist_img, fft_img, radial_img


def make_delta_image(orig: Image.Image, proc: Image.Image, max_size: int = 256):
    """Return (diff_pil, mse, norm_diff) comparing orig vs proc.

    - diff_pil: absolute-difference thumbnail (RGB)
    - mse: mean squared error (float)
    - norm_diff: mean absolute difference normalized to [0..1]
    """
    try:
        # Downscale to reasonable size for cheap diffing
        orig_small = orig.copy()
        proc_small = proc.copy()
        orig_small.thumbnail((max_size, max_size))
        proc_small.thumbnail((max_size, max_size))

        a = np.asarray(orig_small).astype(np.float32)
        b = np.asarray(proc_small).astype(np.float32)
        # Ensure same shape
        if a.shape != b.shape:
            # try to convert proc to orig shape via resize
            proc_small = proc_small.resize(orig_small.size)
            b = np.asarray(proc_small).astype(np.float32)

        diff = np.abs(a - b)
        mse = float(((a - b) ** 2).mean())
        norm_diff = float(diff.mean() / 255.0)

        # Scale diff for visibility
        diff_vis = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff_vis)
        metrics = f"MSE: {mse:.2f}\nMean abs diff (norm): {norm_diff:.4f}"
        return diff_img, mse, metrics
    except Exception as e:
        return None, 0.0, f"delta error: {e}"


def build_interface():
    try:
        import gradio as gr
    except Exception:
        raise RuntimeError("Gradio is not installed. Add 'gradio' to requirements.txt and install it.")

    with gr.Blocks() as demo:
        gr.Markdown("# Image Postprocess — Gradio frontend\nWraps the repository's `process_image` pipeline.")

        with gr.Row():
            inp = gr.Image(type="pil", label="Input image")
            out = gr.Image(type="pil", label="Processed image")

        # Preset selector + save/load
        customs = list(load_custom_presets().keys())
        preset_choices = [*BUILTIN_PRESETS.keys(), *customs]
        with gr.Row():
            preset = gr.Dropdown(choices=preset_choices, value="Preview (no camera sim)", label="Preset")
            preset_name = gr.Textbox(label="Save preset as (name)")
            save_preset_btn = gr.Button("Save preset")
        preset_summary_box = gr.Textbox(value=preset_summary("Default"), label="Preset summary", interactive=False)

        with gr.Row():
            hist_out = gr.Image(type="pil", label="Processed hist")
            fft_out = gr.Image(type="pil", label="Processed FFT")
            radial_out = gr.Image(type="pil", label="Processed radial")
            delta_out = gr.Image(type="pil", label="Diff (abs) thumb")
            delta_metrics = gr.Textbox(label="Delta metrics", interactive=False)

        # Reference images and EXIF output
        with gr.Row():
            awb_ref = gr.Image(type="pil", label="AWB reference (optional)")
            fft_ref = gr.Image(type="pil", label="FFT reference (optional)")
            exif_out = gr.Textbox(label="EXIF (hex)")

        with gr.Row():
            noise_std = gr.Slider(0.0, 0.1, value=0.02, step=0.001, label="Noise STD (fraction)")
            clahe_clip = gr.Slider(0.5, 10.0, value=2.0, step=0.1, label="CLAHE clip")
            tile = gr.Slider(2, 32, value=8, step=1, label="CLAHE tile")

        with gr.Row():
            cutoff = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Fourier cutoff")
            fstrength = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Fourier strength")
            awb = gr.Checkbox(label="Apply AWB (auto white balance)", value=True)

        with gr.Row():
            sim_camera = gr.Checkbox(label="Simulate camera pipeline", value=False)
            lut_file = gr.File(label="Optional LUT (png/npy/cube)")
            lut_strength = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="LUT strength")

        with gr.Row():
            seed = gr.Number(value=None, label="Seed (integer, optional)")
            jpeg_cycles = gr.Slider(1, 5, value=1, step=1, label="JPEG cycles")
            jpeg_qmin = gr.Slider(30, 100, value=88, step=1, label="JPEG quality (min)")

        with gr.Row():
            vignette_strength = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Vignette strength")
            chroma_strength = gr.Slider(0.0, 5.0, value=1.2, step=0.1, label="Chroma strength")
            iso_scale = gr.Slider(0.1, 16.0, value=1.0, step=0.1, label="ISO scale")

        with gr.Row():
            read_noise = gr.Slider(0.0, 50.0, value=2.0, step=0.1, label="Read noise")
            no_no_bayer = gr.Checkbox(label="Disable Bayer (no demosaic)", value=False)

        # Advanced panel for expert parameters
        with gr.Accordion("Advanced parameters (expert)", open=False):
            randomness = gr.Slider(0.0, 0.5, value=0.05, step=0.001, label="Fourier randomness")
            perturb = gr.Slider(0.0, 0.05, value=0.008, step=0.001, label="Perturb magnitude")
            phase_perturb = gr.Slider(0.0, 0.5, value=0.08, step=0.001, label="Phase perturb")
            radial_smooth = gr.Slider(0, 50, value=5, step=1, label="Radial smooth")
            fft_mode = gr.Dropdown(["auto", "ref", "model"], value="auto", label="FFT mode")
            fft_alpha = gr.Slider(0.1, 4.0, value=1.0, step=0.1, label="FFT alpha (1/f)")

        status = gr.Textbox(label="Status", interactive=False)

        def _wrap(preset, inp_img, noise_std, clahe_clip, tile, cutoff, fstrength, awb, sim_camera, lut_file, lut_strength, awb_ref, fft_ref, seed, jpeg_cycles, jpeg_qmin, vignette_strength, chroma_strength, iso_scale, read_noise, no_no_bayer, randomness, perturb, phase_perturb, radial_smooth, fft_mode, fft_alpha):
            jpeg_qmax = 96
            lut_path = lut_file.name if getattr(lut_file, 'name', None) else None

            # Build effective parameters mapping from UI inputs
            params = {
                "noise_std": float(noise_std),
                "clahe_clip": float(clahe_clip),
                "tile": int(tile),
                "cutoff": float(cutoff),
                "fstrength": float(fstrength),
                "awb": bool(awb),
                "sim_camera": bool(sim_camera),
                "lut_file": lut_path,
                "lut_strength": float(lut_strength),
                "awb_ref": awb_ref,
                "fft_ref": fft_ref,
                "seed": int(seed) if (seed is not None and str(seed) != "") else None,
                "jpeg_cycles": int(jpeg_cycles),
                "jpeg_qmin": int(jpeg_qmin),
                "jpeg_qmax": int(jpeg_qmax),
                "vignette_strength": float(vignette_strength),
                "chroma_strength": float(chroma_strength),
                "iso_scale": float(iso_scale),
                "read_noise": float(read_noise),
                "no_no_bayer": bool(no_no_bayer),
                "randomness": float(randomness),
                "perturb": float(perturb),
                "phase_perturb": float(phase_perturb),
                "radial_smooth": int(radial_smooth),
                "fft_mode": fft_mode,
                "fft_alpha": float(fft_alpha),
            }

            # Overlay preset overrides (builtin or custom)
            overrides = get_preset_overrides(preset)
            for k, v in overrides.items():
                params[k] = v

            # Call the pipeline with explicit params
            try:
                result, msg, exif = run_process_with_exif(
                    inp_img,
                    noise_std=params.get("noise_std"),
                    clahe_clip=params.get("clahe_clip"),
                    tile=params.get("tile"),
                    cutoff=params.get("cutoff"),
                    fstrength=params.get("fstrength"),
                    awb=params.get("awb"),
                    sim_camera=params.get("sim_camera"),
                    lut_file=params.get("lut_file"),
                    lut_strength=params.get("lut_strength"),
                    awb_ref=params.get("awb_ref"),
                    fft_ref=params.get("fft_ref"),
                    seed=params.get("seed"),
                    jpeg_cycles=params.get("jpeg_cycles"),
                    jpeg_qmin=params.get("jpeg_qmin"),
                    jpeg_qmax=params.get("jpeg_qmax"),
                    vignette_strength=params.get("vignette_strength"),
                    chroma_strength=params.get("chroma_strength"),
                    iso_scale=params.get("iso_scale"),
                    read_noise=params.get("read_noise"),
                    no_no_bayer=params.get("no_no_bayer"),
                    randomness=params.get("randomness"),
                    perturb=params.get("perturb"),
                    phase_perturb=params.get("phase_perturb"),
                    radial_smooth=params.get("radial_smooth"),
                    fft_mode=params.get("fft_mode"),
                    fft_alpha=params.get("fft_alpha"),
                )
            except Exception as e:
                return None, None, None, None, None, "", f"Processing error: {e}"

            if result is None:
                return None, None, None, None, None, "", msg

            try:
                hist_img, fft_img, radial_img = make_analysis_images(result)
            except Exception as e:
                return result, None, None, None, None, exif, f"Analysis error: {e}"

            # Delta preview
            try:
                diff_img, mse_val, metrics = make_delta_image(inp_img, result)
            except Exception as e:
                diff_img, mse_val, metrics = None, 0.0, f"delta error: {e}"

            return result, hist_img, fft_img, radial_img, diff_img, metrics, exif, msg

        btn = gr.Button("Run")
        btn.click(
            _wrap,
            inputs=[preset, inp, noise_std, clahe_clip, tile, cutoff, fstrength, awb, sim_camera, lut_file, lut_strength, awb_ref, fft_ref, seed, jpeg_cycles, jpeg_qmin, vignette_strength, chroma_strength, iso_scale, read_noise, no_no_bayer, randomness, perturb, phase_perturb, radial_smooth, fft_mode, fft_alpha],
            outputs=[out, hist_out, fft_out, radial_out, delta_out, delta_metrics, exif_out, status],
        )

        def _save_preset(name, preset, noise_std, clahe_clip, tile, cutoff, fstrength, awb, sim_camera, lut_strength, seed, jpeg_cycles, jpeg_qmin, vignette_strength, chroma_strength, iso_scale, read_noise, no_no_bayer, randomness, perturb, phase_perturb, radial_smooth, fft_mode, fft_alpha):
            if not name:
                return "Provide a name to save the preset"
            data = {
                "noise_std": float(noise_std),
                "clahe_clip": float(clahe_clip),
                "tile": int(tile),
                "cutoff": float(cutoff),
                "fstrength": float(fstrength),
                "randomness": float(randomness),
                "perturb": float(perturb),
                "phase_perturb": float(phase_perturb),
                "radial_smooth": int(radial_smooth),
                "jpeg_cycles": int(jpeg_cycles),
                "jpeg_qmin": int(jpeg_qmin),
                "vignette_strength": float(vignette_strength),
                "chroma_strength": float(chroma_strength),
                "iso_scale": float(iso_scale),
                "read_noise": float(read_noise),
                "no_no_bayer": bool(no_no_bayer),
            }
            save_custom_preset(name, data)
            return f"Saved preset: {name}"

        save_preset_btn.click(_save_preset, inputs=[preset_name, preset, noise_std, clahe_clip, tile, cutoff, fstrength, awb, sim_camera, lut_strength, seed, jpeg_cycles, jpeg_qmin, vignette_strength, chroma_strength, iso_scale, read_noise, no_no_bayer, randomness, perturb, phase_perturb, radial_smooth, fft_mode, fft_alpha], outputs=[preset_summary_box])

        def _update_summary(selected):
            return preset_summary(selected)

        preset.change(_update_summary, inputs=[preset], outputs=[preset_summary_box])

    return demo


if __name__ == "__main__":
    iface = build_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860)
