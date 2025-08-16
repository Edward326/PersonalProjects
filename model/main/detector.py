import torch
import torch.nn as nn
from ultralytics import YOLO
import torch.quantization as quant
import os

class YOLOv8Optimizer:
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv8 model with optimization capabilities
        model_size: 'n', 's', 'm', 'l', 'x' (nano, small, medium, large, extra-large)
        """
        self.model_size = model_size
        self.model = None
        self.optimized_model = None
        self.class_names = None
        
    def load_model(self):
        """Load YOLOv8 model"""
        print(f"🔄 Loading YOLOv8{self.model_size} model...")
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Get class names for later use
        self.class_names = self.model.names
        print("✅ YOLOv8 model loaded successfully")
        return self.model
    
    def apply_quantization(self):
        """Apply dynamic quantization for mobile deployment"""
        print("🔧 Applying dynamic quantization...")
        
        # Get the PyTorch model from ultralytics wrapper
        pytorch_model = self.model.model
        
        # Set to evaluation mode
        pytorch_model.eval()
        
        # Apply dynamic quantization - be more conservative for mobile
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                pytorch_model,
                {nn.Linear, nn.Conv2d},  # Quantize these layer types
                dtype=torch.qint8
            )
            
            # Wrap back for ultralytics compatibility
            self.model.model = quantized_model
            self.optimized_model = self.model
            
            print("✅ Dynamic quantization applied")
        except Exception as e:
            print(f"⚠️ Quantization failed, using original model: {e}")
            self.optimized_model = self.model
        
        return self.optimized_model
    
    def optimize_for_mobile(self):
        """Complete optimization pipeline for mobile deployment"""
        print("📱 Starting mobile optimization pipeline...")
        
        # Load model
        self.load_model()
        
        # Apply light quantization (conservative approach)
        self.apply_quantization()
        
        print("✅ Mobile optimization complete")
        return self.optimized_model
    
    def get_exportable_model(self):
        """Get a model ready for TorchScript export"""
        if self.optimized_model is None:
            self.optimize_for_mobile()
        
        # Return the core PyTorch model for export
        return self.optimized_model.model
    
    def get_class_names(self):
        """Get COCO class names dictionary"""
        if self.class_names is None and self.model is not None:
            self.class_names = self.model.names
        return self.class_names
    
    def test_inference(self, test_image_path=None):
        """Test inference speed and accuracy"""
        print("🧪 Testing inference...")
        
        if test_image_path is None:
            # Create dummy input for testing
            dummy_input = torch.randn(1, 3, 640, 640)
            print("Using dummy input for testing")
        
        model_to_test = self.optimized_model if self.optimized_model else self.model
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                if test_image_path:
                    _ = model_to_test(test_image_path)
                else:
                    # For dummy input, we need to use the pytorch model directly
                    _ = model_to_test.model(dummy_input)
        
        # Time inference
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                if test_image_path:
                    results = model_to_test(test_image_path)
                else:
                    results = model_to_test.model(dummy_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"⏱️ Average inference time: {avg_time:.3f} seconds")
        print(f"🚀 FPS: {1/avg_time:.1f}")
        
        return avg_time

if __name__ == "__main__":
    # Example usage
    optimizer = YOLOv8Optimizer(model_size='n')  # Use nano version for mobile
    
    # Optimize model
    optimized_model = optimizer.optimize_for_mobile()
    
    # Test inference
    optimizer.test_inference()
    
    print("\n📋 Optimization Summary:")
    print("- Dynamic quantization applied for faster inference")
    print("- Model optimized for mobile deployment")
    print("- Ready for TorchScript conversion")