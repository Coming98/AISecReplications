"""
Splits a directory of CelebA-HQ images into train/val/test subfolders.

CelebA-HQ does not ship its own train/val/test partition, so we recover one by:
  1. Mapping each CelebA-HQ image's index to its original CelebA filename, via
     CelebA-HQ-to-CelebA-mapping.txt.
  2. Looking up that CelebA filename's split (0=train, 1=val, 2=test) in
     CelebA's official list_eval_partition.txt.

Input images are expected to be named by their bare CelebA-HQ index
(e.g. "0.jpg", "000003.jpg" - zero-padding does not matter).

Usage:
    python scripts/dataset_split.py \
        --celeba_hq_dir ./dataset/image/celeba-hq_256 \
        --mapping_file ./inputs/CelebA-HQ-to-CelebA-mapping.txt \
        --partition_file ./inputs/list_eval_partition.txt

Output defaults to "<celeba_hq_dir>_splited" with train/val/test subfolders,
next to the input directory.
"""
import argparse
import os
import shutil

SPLIT_NAMES = {0: 'train', 1: 'val', 2: 'test'}


def load_hq_to_celeba_mapping(mapping_file):
    """Parses CelebA-HQ-to-CelebA-mapping.txt into {hq_idx: orig_celeba_filename}."""
    mapping = {}
    with open(mapping_file, 'r') as f:
        next(f)  # header: idx orig_idx orig_file
        for line in f:
            fields = line.split()
            if not fields:
                continue
            hq_idx, _orig_idx, orig_file = fields[0], fields[1], fields[2]
            mapping[int(hq_idx)] = orig_file
    return mapping


def load_celeba_partition(partition_file):
    """Parses list_eval_partition.txt into {celeba_filename: split_id}."""
    partition = {}
    with open(partition_file, 'r') as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            filename, split_id = fields[0], int(fields[1])
            partition[filename] = split_id
    return partition


def hq_index_from_filename(filename):
    """Recovers the integer CelebA-HQ index from a bare-index filename, e.g. '000003.jpg' -> 3."""
    stem, _ext = os.path.splitext(filename)
    try:
        return int(stem)
    except ValueError:
        return None


def place_file(src_path, dst_path, op):
    if op == 'copy':
        shutil.copy2(src_path, dst_path)
    elif op == 'move':
        shutil.move(src_path, dst_path)
    elif op == 'symlink':
        os.symlink(os.path.abspath(src_path), dst_path)
    else:
        raise ValueError(f'Unknown op: {op}')


def split_dataset(celeba_hq_dir, hq_to_celeba, celeba_partition, output_dir, op, dry_run):
    for split_name in SPLIT_NAMES.values():
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    counts = {split_name: 0 for split_name in SPLIT_NAMES.values()}
    skipped = []

    for filename in sorted(os.listdir(celeba_hq_dir)):
        src_path = os.path.join(celeba_hq_dir, filename)
        if not os.path.isfile(src_path):
            continue

        hq_idx = hq_index_from_filename(filename)
        if hq_idx is None or hq_idx not in hq_to_celeba:
            skipped.append((filename, 'no CelebA-HQ -> CelebA mapping entry'))
            continue

        celeba_filename = hq_to_celeba[hq_idx]
        split_id = celeba_partition.get(celeba_filename)
        if split_id not in SPLIT_NAMES:
            skipped.append((filename, f'no partition entry for {celeba_filename}'))
            continue

        split_name = SPLIT_NAMES[split_id]
        dst_path = os.path.join(output_dir, split_name, filename)
        if not dry_run:
            place_file(src_path, dst_path, op)
        counts[split_name] += 1

    return counts, skipped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--celeba_hq_dir', required=True,
                         help='Directory of CelebA-HQ images named by their bare HQ index.')
    parser.add_argument('--mapping_file', required=True,
                         help='Path to CelebA-HQ-to-CelebA-mapping.txt.')
    parser.add_argument('--partition_file', required=True,
                         help='Path to CelebA\'s list_eval_partition.txt.')
    parser.add_argument('--output_dir', default=None,
                         help='Defaults to "<celeba_hq_dir>_splited" next to the input directory.')
    parser.add_argument('--op', choices=['copy', 'move', 'symlink'], default='copy',
                         help='How to place images into the split folders (default: copy).')
    parser.add_argument('--dry_run', action='store_true',
                         help='Only report split counts, without creating/writing any files.')
    return parser.parse_args()


def main():
    args = parse_args()
    celeba_hq_dir = os.path.normpath(args.celeba_hq_dir)
    output_dir = args.output_dir or f'{celeba_hq_dir}_splited'

    hq_to_celeba = load_hq_to_celeba_mapping(args.mapping_file)
    celeba_partition = load_celeba_partition(args.partition_file)
    counts, skipped = split_dataset(
        celeba_hq_dir, hq_to_celeba, celeba_partition, output_dir, args.op, args.dry_run
    )

    print(f'{"[DRY RUN] " if args.dry_run else ""}Output directory: {output_dir}')
    print(f"Total images processed: {sum(counts.values())}")
    for split_name, count in counts.items():
        print(f'  {split_name}: {count} images')

    if skipped:
        print(f'Skipped {len(skipped)} images:')
        for filename, reason in skipped[:20]:
            print(f'  {filename}: {reason}')
        if len(skipped) > 20:
            print(f'  ... and {len(skipped) - 20} more')


if __name__ == '__main__':
    main()