import torch
from ultralytics import YOLO
import os

class YOLOClassifierConverter:
    def __init__(self, model_path):
        """
        model_path: path to YOLO .pt model (e.g., 'yolov8n.pt')
        """
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load YOLO model"""
        print(f"📄 Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            print("✅ YOLO model loaded successfully")
            return self.model
        except Exception as e:
            print(f"❌ Failed to load YOLO model:\nexception msg:{e}")
            self.model = None
            self.class_names = None
            return None
        
    def export_to_onnx_simple(self, save_path):
        """
        Export to ONNX WITHOUT NMS (most reliable)
        Output shape: [1, 84, 8400]
        """
        print("🔧 Exporting YOLO to ONNX (simple mode)...")
        
        try:
            exported_file = self.model.export(
                format="onnx",
                imgsz=640,
                half=True,          # FP16 for speed
                dynamic=False,      # Fixed size
                simplify=True,      # Optimize
                opset=12,
                nms=False,          # No NMS (do in Java)
            )
            
            if exported_file and os.path.exists(exported_file):
                if save_path != exported_file:
                    import shutil
                    shutil.move(exported_file, save_path)
                
                print(f"✅ Export successful: {save_path}")
                return save_path
            else:
                print(f"❌ Export failed")
                return None
                
        except Exception as e:
            print(f"❌ Export error: {e}")
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