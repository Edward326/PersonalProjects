from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    print(f"📄 Loading YOLO base model {args.model}")
    model = YOLO(args.model)

    print("🚀 Starting training...")
    model.train(
        data=args.yaml,
        epochs=args.epochs,
        lr0=args.lr,
        batch=args.batch,
        optimizer="AdamW",  # good default
        device=0
    )

    model.save("indoor_trained.pt")
    print("✅ Training finished and model saved")

if __name__ == "__main__":
    main()
