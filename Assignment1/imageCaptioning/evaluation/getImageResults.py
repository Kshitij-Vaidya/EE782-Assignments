
import json
import os
from PIL import Image, ImageDraw, ImageFont

# Load generated captions
with open("../outputs/lstm_resnet18_captions.json", "r") as f:
    data = json.load(f)

# Load ground truth annotations
with open("../rsicdDataset/testAnnotations.json", "r") as f:
    gt_data = json.load(f)

# Build a mapping from filename to ground truth captions
gt_map = {item["filename"]: item["captions"] for item in gt_data}

# Font settings (adjust path if needed)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

output_dir = "../outputs/captioned_images"
os.makedirs(output_dir, exist_ok=True)

for entry in data:
    img_path = entry["filepath"]
    caption = entry["caption"]
    # Get ground truth captions from mapping
    filename = os.path.basename(img_path)
    ground_truth_list = gt_map.get(filename, [])
    ground_truth = ground_truth_list[0] if ground_truth_list else "Ground Truth: N/A"
    # Load image
    img = Image.open(img_path).convert("RGB")
    # Create a new image with extra space for text
    width, height = img.size
    new_img = Image.new("RGB", (width, height + 80), (255, 255, 255))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    # Draw ground truth and generated caption
    draw.text((10, height + 10), f"GT: {ground_truth}", fill=(0, 128, 0), font=font)
    draw.text((10, height + 40), f"Caption: {caption}", fill=(0, 0, 0), font=font)
    # Save image
    out_path = os.path.join(output_dir, os.path.basename(img_path).replace(".jpg", "_captioned.png"))
    new_img.save(out_path)
    print(f"Saved: {out_path}")
