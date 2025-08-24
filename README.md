---
title: Deepfake Detection Bypass Utility
emoji: 👁
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.35.0
app_file: app_gradio.py
pinned: true
short_description: DF detection circumvention utility refactored for HF Spaces.
---


# Image Detection Bypass Utility [Gradio Refactor]

Refactored Gradio app for HF Spaces of [Image Detection Bypass](https://github.com/PurinNyova/Image-Detection-Bypass-Utility), self-proclaimed app for the "Circumvention of AI Detection."

Strictly for research purposes -- please refer to the original repository for more in-depth documentation and up-to-date code. 

This refactored repository will not be maintained.

---

## Screenshot

![image/png](https://cdn-uploads.huggingface.co/production/uploads/639daf827270667011153fbc/zMAmg93CH66zgMkgxAsez.png)

---

## Parameters / Controls → `args` mapping

When you click **Run**, the GUI builds a lightweight argument namespace (similar to a `SimpleNamespace`) and passes it to the worker. Below are the important mappings used by the GUI (so you know what your `process_image` should expect):

- `args.noise_std` — Gaussian noise STD (fraction of 255)  
- `args.clahe_clip` — CLAHE clip limit  
- `args.tile` — CLAHE tile size  
- `args.cutoff` — Fourier cutoff (0.01–1.0)  
- `args.fstrength` — Fourier strength (0–1)  
- `args.phase_perturb` — phase perturbation STD (radians)  
- `args.randomness` — Fourier randomness factor  
- `args.perturb` — small pixel perturbations  
- `args.fft_mode` — one of `auto`, `ref`, `model`  
- `args.fft_alpha` — alpha exponent for 1/f model (used when `fft_mode=='model'`)  
- `args.radial_smooth` — radial smoothing bins for spectrum matching  
- `args.jpeg_cycles` — number of lossy JPEG encode/decode cycles (camera sim)  
- `args.jpeg_qmin`, `args.jpeg_qmax` — JPEG quality range used by camera sim  
- `args.vignette_strength` — vignette intensity (0–1)  
- `args.chroma_strength` — chromatic aberration strength (pixels)  
- `args.iso_scale` — exposure multiplier (camera sim)  
- `args.read_noise` — read noise in DN (camera sim)  
- `args.hot_pixel_prob` — probability of hot pixels (camera sim)  
- `args.banding_strength` — banding strength  
- `args.motion_blur_kernel` — motion blur kernel size  
- `args.seed` — integer seed or `None` when seed==0 in UI  
- `args.sim_camera` — bool: run camera simulation path  
- `args.no_no_bayer` — toggles Bayer/demosaic (True = enable RGGB demosaic)  
- `args.fft_ref` — path to reference image (string) or `None`
- `args.ref`— path to auto white-balance reference image (string) or `None`

> **Tip:** Your `process_image(inpath, outpath, args)` should be tolerant of missing keys (use `getattr(args, 'name', default)`), or accept the same `SimpleNamespace` object the GUI builds.

---

## Gradio / Hugging Face Spaces

This repository includes a lightweight Gradio front-end (`app_gradio.py`) that wraps the existing
`process_image(inpath, outpath, args)` pipeline. The Gradio app is suitable for local testing and
for deployment to Hugging Face Spaces (Gradio-backed web apps).

### Quick local run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the Gradio app:

```bash
python3 app_gradio.py
```

Open http://localhost:7860 in your browser. The UI saves the uploaded image to a temporary file,
calls the existing `process_image` pipeline, and returns the processed image.

### Deploying to Hugging Face Spaces

1. Ensure the following are present at the repository root:
     - `app_gradio.py` (the Gradio entrypoint)
     - `requirements.txt` (must include `gradio` and any other runtime deps)

2. Push the repository to a new Space on Hugging Face (create a new Space and connect this repo or
     push to the Space's Git remote). Spaces will automatically run the Gradio app.

Notes & tips for Spaces:
- Keep default upload/processing sizes modest to avoid long CPU usage in the free tier.
- If your pipeline uses optional packages (OpenCV, piexif, etc.), make sure they are listed in
    `requirements.txt` so Spaces installs them.
- If processing is slow, consider reducing default image size or exposing fewer parameters to the
    main UI and keeping advanced controls hidden in an "Advanced" section.

### Troubleshooting
- If Gradio is not installed, `app_gradio.py` will raise an error; add `gradio` to `requirements.txt`.
- Any import errors from `image_postprocess` will surface when calling the app; run the smoke test
    (`python3 test_smoke_gradio.py`) locally to validate imports and pipeline execution before pushing.

---

## License
MIT — free to use and adapt. Please include attribution if you fork or republish.

---
