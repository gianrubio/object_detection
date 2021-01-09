from PIL import Image
import os, sys
from pathlib import Path

cur_path = Path(__file__).parent


path = str((cur_path / f"../images/train/").resolve())
dirs = os.listdir(path)
final_size = 200


def resize_aspect_fit():
    for item in dirs:
        if item == ".DS_Store":
            continue
        file = os.path.join(path, item)
        if os.path.isfile(file) and item.lower().endswith(".jpg"):
            im = Image.open(file)
            f, e = os.path.splitext(file)
            size = im.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (final_size, final_size))
            new_im.paste(
                im,
                (
                    (final_size - new_image_size[0]) // 2,
                    (final_size - new_image_size[1]) // 2,
                ),
            )
            new_im.save(f + ".jpg", "JPEG", quality=90)


resize_aspect_fit()