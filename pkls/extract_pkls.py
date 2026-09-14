"""Extract the dataset cache archives shipped in ./pkls/.

The preprocessed caches are stored as one zip per file so that each archive
stays under GitHub's 100 MB per-file limit. Run this script once before the
first training/evaluation run:

    python pkls/extract_pkls.py

The archives are named with a leading `._`, which makes them hidden files, so
a plain `unzip *.zip` does not match them - this script globs them explicitly.

Options:
    --force   re-extract even if the .pkl already exists
    --list    only report what would be extracted, write nothing
"""

import argparse
import os
import zipfile

PKL_DIR = os.path.dirname(os.path.abspath(__file__))


def human(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='re-extract archives whose .pkl is already present')
    parser.add_argument('--list', action='store_true', dest='list_only',
                        help='list the archives and their contents without extracting')
    args = parser.parse_args()

    archives = sorted(f for f in os.listdir(PKL_DIR) if f.endswith('.zip'))
    if not archives:
        print(f'No .zip archives found in {PKL_DIR}')
        return

    print(f'Found {len(archives)} archive(s) in {PKL_DIR}\n')

    extracted, skipped = 0, 0
    for name in archives:
        archive_path = os.path.join(PKL_DIR, name)
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            total = sum(m.file_size for m in members)

            if args.list_only:
                print(f'{name} -> {", ".join(m.filename for m in members)} '
                      f'({human(total)} extracted)')
                continue

            missing = [m for m in members
                       if not os.path.exists(os.path.join(PKL_DIR, m.filename))]
            if not missing and not args.force:
                print(f'[skip] {name} (already extracted)')
                skipped += 1
                continue

            print(f'[extract] {name} -> {human(total)} ... ', end='', flush=True)
            zf.extractall(PKL_DIR)
            print('done')
            extracted += 1

    if not args.list_only:
        print(f'\nExtracted {extracted} archive(s), skipped {skipped}.')
        if skipped and not args.force:
            print('Use --force to overwrite the existing .pkl files.')


if __name__ == '__main__':
    main()
