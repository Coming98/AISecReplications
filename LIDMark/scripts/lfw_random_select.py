"""
Randomly selects {count} identities from an LFW-style directory (one subfolder
per identity), picking one random image per selected identity, and places the
result in a flat "lfw_{count}" output folder.

Usage:
    python scripts/lfw_random_select.py \
        --lfw_dir ./dataset/image/lfw \
        --count 2000 \
        --seed 42
"""
import argparse
import os
import random
import shutil


def list_identities(lfw_dir):
    return sorted(
        name for name in os.listdir(lfw_dir)
        if os.path.isdir(os.path.join(lfw_dir, name))
    )


def select_one_image_per_identity(lfw_dir, identities, rng):
    selected = []
    for identity in identities:
        identity_dir = os.path.join(lfw_dir, identity)
        images = sorted(
            filename for filename in os.listdir(identity_dir)
            if os.path.isfile(os.path.join(identity_dir, filename))
        )
        if not images:
            continue
        chosen = rng.choice(images)
        selected.append((identity, chosen))
    return selected


def place_file(src_path, dst_path, op):
    if op == 'copy':
        shutil.copy2(src_path, dst_path)
    elif op == 'move':
        shutil.move(src_path, dst_path)
    elif op == 'symlink':
        os.symlink(os.path.abspath(src_path), dst_path)
    else:
        raise ValueError(f'Unknown op: {op}')


def build_output(lfw_dir, selected, output_dir, op, dry_run):
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    for identity, filename in selected:
        src_path = os.path.join(lfw_dir, identity, filename)
        dst_path = os.path.join(output_dir, filename)
        if not dry_run:
            place_file(src_path, dst_path, op)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lfw_dir', required=True,
                         help='LFW directory containing one subfolder per identity.')
    parser.add_argument('--count', type=int, required=True,
                         help='Number of identities to randomly select.')
    parser.add_argument('--output_dir', default=None,
                         help='Defaults to "<lfw_dir>_{count}" next to the input directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible selection.')
    parser.add_argument('--op', choices=['copy', 'move', 'symlink'], default='copy',
                         help='How to place images into the output folder (default: copy).')
    parser.add_argument('--dry_run', action='store_true',
                         help='Only report the selection, without creating/writing any files.')
    return parser.parse_args()


def main():
    args = parse_args()
    lfw_dir = os.path.normpath(args.lfw_dir)
    output_dir = args.output_dir or f'{lfw_dir}_{args.count}'

    identities = list_identities(lfw_dir)
    if args.count > len(identities):
        raise ValueError(f'Requested {args.count} identities but only {len(identities)} are available.')

    rng = random.Random(args.seed)
    sampled_identities = rng.sample(identities, args.count)
    selected = select_one_image_per_identity(lfw_dir, sampled_identities, rng)

    build_output(lfw_dir, selected, output_dir, args.op, args.dry_run)

    print(f'{"[DRY RUN] " if args.dry_run else ""}Output directory: {output_dir}')
    print(f'Selected {len(selected)} images from {len(identities)} available identities.')


if __name__ == '__main__':
    main()