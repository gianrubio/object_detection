import os
from labels import generate_file_labels
from PIL import Image
# generate labels based in a jpg image
file_path = "/Users/grubio/Downloads/bounding_boxes/patagonia_lata_475ml"
label_name = "patagonia_lata_475ml"

for path, dirs, files in os.walk(file_path):
    for file in files:
        if not file.endswith(".jpg"):
            continue
        file_name = file.split(".jpg")[0]
        im = Image.open(os.path.join(file_path, file))
        w, h = im.size
        generate_file_labels(
            label_name, f"{file_name}", file_path, w, h, 0, 0, w, h
        )
        # print(f"rename from {os.path.join(path, file)} to {os.path.join(path, file.replace(' ', ''))}")
