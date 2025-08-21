import argparse
import os

def make_yaml(dataset_dir, classes_file, output_yaml):
    # Read classes
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"❌ Classes file not found: {classes_file}")
    
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    # Build YAML content
    yaml_str = "# Dataset root directory relative to the YOLO directory\n"
    yaml_str += f"path: {dataset_dir}\n\n"
    yaml_str += "# Train/val/test sets\n"
    yaml_str += "train: images/train\n"
    yaml_str += "val: images/val\n"
    yaml_str += "test: images/test # optional\n\n"
    yaml_str += "# Classes\n"
    yaml_str += "names:\n"
    for i, cls in enumerate(classes):
        yaml_str += f"  {i}: {cls}\n"

    # Save YAML
    with open(output_yaml, "w") as f:
        f.write(yaml_str)

    print(f"✅ YAML file saved at {output_yaml}")
    print(f"📦 {len(classes)} classes included.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO YAML generator")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to merged dataset dir")
    parser.add_argument("--classes_file", type=str, required=True, help="Path to classes.txt file")
    parser.add_argument("--output_yaml", type=str, default="dataset.yaml", help="Output YAML file path")
    args = parser.parse_args()

    make_yaml(args.dataset_dir, args.classes_file, args.output_yaml)