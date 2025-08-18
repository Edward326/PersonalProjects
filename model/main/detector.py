import torch
from ultralytics import YOLO
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

class YOLOv8Optimizer:
    def __init__(self, model_path, save_path):
        """
        model_path: path to YOLO .pt model (e.g., 'yolov8n.pt')
        save_path: final export path for .ptl file (e.g., '../saved/detector/yolov8n_mobile.ptl')
        """
        self.model_path = model_path
        self.save_path = save_path
        self.model = None
        self.class_names = None

    def load_model(self):
        """Load YOLOv8 model"""
        print(f"📄 Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            print("✅ YOLOv8 model loaded successfully")
            return self.model
        except Exception as e:
            print(f"❌ Failed to load YOLOv8 model: {e}")
            self.model = None
            self.class_names = None
        
    def export_to_ptl(self):
        """Export YOLOv8 to TorchScript + optimize for mobile -> save .ptl"""
        if self.model is None:
            self.load_model()

        print("🔧 Exporting to TorchScript...")
        try:
            exported_file = self.model.export(
                format="torchscript",
                optimize=True,
                half=False,
                dynamic=False,
                simplify=True
            )

            if not exported_file or not os.path.exists(exported_file):
                print(f"❌ Export failed: {exported_file} not found")
                return None

            # Load TorchScript model
            ts_model = torch.jit.load(exported_file)
            ts_model.eval()

            # Optimize for mobile
            print("📱 Optimizing TorchScript for mobile...")
            optimized_model = optimize_for_mobile(ts_model)

            # Save as Lite Interpreter .ptl
            optimized_model._save_for_lite_interpreter(self.save_path)
            print(f"✅ Export complete! Saved at {self.save_path}")
            os.remove(self.model_path);os.remove(os.path.splitext(self.model_path)[0]+'.torchscript')
            return self.save_path

        except Exception as e:
            print(f"❌ Export to .ptl failed: {e}")
            return None

    def get_class_names(self):
        if self.class_names is None and self.model is not None:
            self.class_names = self.model.names
        return self.class_names

    def save_class_names(self, output_path):
        """Save COCO class names to file"""
        class_names = self.get_class_names()
        if class_names:
            with open(output_path, "w") as f:
                for idx, name in class_names.items():
                    f.write(f"{idx},{name}\n")
            print(f"✅ Class names saved to {output_path}")
            return output_path
        return None