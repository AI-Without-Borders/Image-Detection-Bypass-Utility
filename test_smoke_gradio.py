from PIL import Image
import io
import os

# Import the run_process function from the Gradio wrapper
from app_gradio import run_process

# Create a small synthetic test image (RGB gradient)
img = Image.new('RGB', (128, 128))
for y in range(128):
    for x in range(128):
        img.putpixel((x, y), (int(x*2), int(y*2), int((x+y)/2)))

out_img, status = run_process(img)
print('Status:', status)
if out_img is not None:
    out_path = '/tmp/test_gradio_output.jpg'
    out_img.save(out_path)
    print('Wrote output to', out_path)
else:
    print('No output image generated')
