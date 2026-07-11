import argparse
import os
import sys
import requests
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm

def download_file(url, output_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(output_path, 'wb') as f, tqdm(
        desc=f"⬇️ Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    if total_size != 0 and os.path.getsize(output_path) != total_size:
        raise Exception("❌ Download incomplete or corrupted")
    print(f"✅ Download complete: {output_path}")

def extract_archive(archive_path, extract_dir):
    """Extract zip/tar archives"""
    print(f"📂 Extracting {archive_path} ...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)
    elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_dir)
    else:
        raise Exception(f"❌ Unsupported archive format: {archive_path}")
    print(f"✅ Extracted to {extract_dir}")

def remap_labels(ann_dir, mapping_file):
    """Remap YOLO labels according to JSON mapping"""
    if not os.path.exists(ann_dir):
        raise Exception(f"❌ Annotation folder not found: {ann_dir}")

    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    print(f"🔧 Remapping labels in {ann_dir} using {mapping_file} ...")

    for txt_file in Path(ann_dir).rglob("*.txt"):
        new_lines = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = parts[0]
                if cls_id in mapping:
                    cls_id = str(mapping[cls_id])
                new_line = " ".join([cls_id] + parts[1:])
                new_lines.append(new_line)
        with open(txt_file, "w") as f:
            f.write("\n".join(new_lines))

    print(f"✅ Label remapping done for {ann_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download + prepare dataset for YOLO")
    parser.add_argument("--dataset_url", type=str, required=True, help="URL to dataset archive")
    parser.add_argument("--ann_loc", type=str, required=True, help="Path to annotation folder inside dataset")
    parser.add_argument("--format", type=str, required=True, help="JSON mapping file: {old_label: new_label}")
    parser.add_argument("--output", type=str, default="./datasets", help="Output root directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    archive_path = os.path.join(args.output, os.path.basename(args.dataset_url))
    dataset_dir = os.path.join(args.output, Path(os.path.basename(args.dataset_url)).stem)

    # 1. Download
    if not os.path.exists(archive_path):
        download_file(args.dataset_url, archive_path)
    else:
        print(f"ℹ️ Archive already exists: {archive_path}")

    # 2. Extract
    if not os.path.exists(dataset_dir):
        extract_archive(archive_path, args.output)
    else:
        print(f"ℹ️ Dataset already extracted: {dataset_dir}")

    # 3. Remap labels
    ann_folder = os.path.join(dataset_dir, args.ann_loc)
    remap_labels(ann_folder, args.format)

if __name__ == "__main__":
    main()