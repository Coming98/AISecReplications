"""
Center-crops (if needed) and resizes every image in a folder to a fixed
square resolution (e.g. 128 or 256), writing the results to an output folder.

Usage:
    python scripts/resize_dataset.py \
        --input_dir ./raw/celeba-hq \
        --output_dir ./dataset/image/celeba-hq_256 \
        --resolution 256
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from tqdm import tqdm

JPEG_SAVE_QUALITY = 95


def center_crop_to_square(img):
    width, height = img.size
    if width == height:
        return img
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def resize_image(src_path, dst_path, resolution):
    img = Image.open(src_path).convert('RGB')
    img = center_crop_to_square(img)
    img = img.resize((resolution, resolution), Image.LANCZOS)
    save_kwargs = {'quality': JPEG_SAVE_QUALITY} if dst_path.lower().endswith(('.jpg', '.jpeg')) else {}
    img.save(dst_path, **save_kwargs)


def _resize_one(args):
    src_path, dst_path, resolution = args
    resize_image(src_path, dst_path, resolution)


def resize_directory(input_dir, output_dir, resolution, num_workers):
    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(
        filename for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename))
    )
    tasks = [
        (os.path.join(input_dir, filename), os.path.join(output_dir, filename), resolution)
        for filename in filenames
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_resize_one, tasks), total=len(tasks), desc=f'Resizing to {resolution}x{resolution}'))

    return len(tasks)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True, help='Folder of source images.')
    parser.add_argument('--output_dir', required=True, help='Folder to write resized images to.')
    parser.add_argument('--resolution', type=int, required=True, help='Target square resolution, e.g. 128 or 256.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                         help='Number of worker processes (default: all CPU cores).')
    return parser.parse_args()


def main():
    args = parse_args()
    count = resize_directory(args.input_dir, args.output_dir, args.resolution, args.num_workers)
    print(f'Resized {count} images to {args.resolution}x{args.resolution} -> {args.output_dir}')


if __name__ == '__main__':
    main()