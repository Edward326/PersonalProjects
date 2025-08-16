import torch
import json
from torch.utils.mobile_optimizer import optimize_for_mobile
import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
# Adjust imports based on your file structure
from main.detector import YOLOv8Optimizer
from main.captioner import Gemma3Optimizer

class ModelSaver:
    def __init__(self, export_dir='../saved/PyTMobile'):
        """
        Initializes the ModelSaver class and creates necessary directories.
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        
        # Create subdirectories for each model
        self.yolo_dir = os.path.join(export_dir, 'yolo')
        self.gemma_dir = os.path.join(export_dir, 'gemma')
        os.makedirs(self.yolo_dir, exist_ok=True)
        os.makedirs(self.gemma_dir, exist_ok=True)
    
    def create_simple_yolo_wrapper(self, pytorch_model):
        """
        Create a simple wrapper for YOLOv8 that's easier to trace
        """
        class SimpleYOLOWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Just return raw model output for mobile
                return self.model(x)
        
        return SimpleYOLOWrapper(pytorch_model)
    
    def export_yolo_model(self):
        """
        Exports the optimized YOLOv8 model for mobile deployment using TorchScript.
        """
        print("\n🔥 Exporting YOLOv8 for mobile...")
        
        # Initialize and optimize YOLOv8
        yolo_optimizer = YOLOv8Optimizer(model_size='n')  # Use nano for mobile
        optimized_yolo = yolo_optimizer.optimize_for_mobile()
        
        if optimized_yolo is None:
            print("❌ YOLOv8 optimization failed")
            return None
        
        try:
            # Get the PyTorch model from ultralytics wrapper
            pytorch_model = optimized_yolo.model
            
            # Set to evaluation mode
            pytorch_model.eval()
            
            # Create a simple wrapper to avoid complex tracing issues
            wrapped_model = self.create_simple_yolo_wrapper(pytorch_model)
            wrapped_model.eval()
            
            # Create a dummy input tensor
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Use torch.jit.trace for the wrapped model
            print("🔍 Tracing YOLOv8 model...")
            
            with torch.no_grad():
                traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
            
            # Apply mobile optimizations
            print("🔧 Applying mobile optimizations...")
            optimized_traced_model = optimize_for_mobile(traced_model)

            # Define export path
            export_path = os.path.join(self.yolo_dir, 'yolov8n_mobile.ptl')
            
            # Save the optimized model
            optimized_traced_model._save_for_lite_interpreter(export_path)
            
            # Save class names for Android app
            class_names_path = os.path.join(self.yolo_dir, 'class_names.json')
            class_names = yolo_optimizer.get_class_names()
            with open(class_names_path, 'w') as f:
                json.dump(class_names, f, indent=2)
            
            print(f"✅ YOLOv8 model successfully exported to {export_path}")
            print(f"✅ Class names saved to {class_names_path}")
            
            return export_path
            
        except Exception as e:
            print(f"❌ YOLOv8 export failed: {e}")
            # Try alternative export method
            try:
                print("🔄 Trying alternative export method...")
                return self.export_yolo_alternative(yolo_optimizer)
            except Exception as e2:
                print(f"❌ Alternative export also failed: {e2}")
                return None

    def export_yolo_alternative(self, yolo_optimizer):
        """Alternative YOLO export method"""
        try:
            # Use ultralytics built-in export
            model = yolo_optimizer.model
            export_path = os.path.join(self.yolo_dir, 'yolov8n.torchscript')
            
            # Export using ultralytics
            model.export(format='torchscript', optimize=True)
            
            # Move the exported file
            import shutil
            source_path = f'yolov8{yolo_optimizer.model_size}.torchscript'
            if os.path.exists(source_path):
                shutil.move(source_path, export_path)
                print(f"✅ YOLOv8 exported using alternative method to {export_path}")
                return export_path
            else:
                return None
        except Exception as e:
            print(f"❌ Alternative export failed: {e}")
            return None

    def create_simple_text_generator(self, model, tokenizer):
        """
        Create a simplified text generator for mobile deployment
        """
        class SimpleTextGenerator(torch.nn.Module):
            def __init__(self, model, tokenizer):
                super().__init__()
                self.model = model
                self.vocab_size = len(tokenizer)
                self.pad_token_id = tokenizer.pad_token_id
                self.eos_token_id = tokenizer.eos_token_id
            
            def forward(self, input_ids, attention_mask=None):
                # Simplified forward pass for mobile
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
        
        return SimpleTextGenerator(model, tokenizer)

    def export_gemma_model(self):
        """
        Exports the optimized Gemma3 model for mobile deployment.
        Note: This is challenging for large language models and may not work on all devices.
        """
        print("\n🔥 Attempting to export language model for mobile...")

        # Initialize and optimize Gemma3
        gemma_optimizer = Gemma3Optimizer()
        result = gemma_optimizer.optimize_for_mobile()

        if result is None or result[0] is None:
            print("❌ Language model optimization failed")
            return None

        try:
            optimized_model, tokenizer = result
            
            # Set to evaluation mode
            optimized_model.eval()
            
            # Create simplified wrapper
            simple_generator = self.create_simple_text_generator(optimized_model, tokenizer)
            simple_generator.eval()
            
            # Create example inputs for tracing
            input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Example token IDs
            attention_mask = torch.ones_like(input_ids)
            
            print("🔍 Tracing language model (this may take a while)...")
            
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    simple_generator, 
                    (input_ids, attention_mask),
                    strict=False
                )
            
            # Apply mobile optimizations
            print("🔧 Applying mobile optimizations...")
            optimized_traced_model = optimize_for_mobile(traced_model)

            # Define export path
            export_path = os.path.join(self.gemma_dir, 'text_generator_mobile.ptl')

            # Save the optimized model
            optimized_traced_model._save_for_lite_interpreter(export_path)
            
            # Save tokenizer info
            tokenizer_info = {
                'vocab_size': len(tokenizer),
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'bos_token_id': getattr(tokenizer, 'bos_token_id', None)
            }
            
            tokenizer_path = os.path.join(self.gemma_dir, 'tokenizer_info.json')
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer_info, f, indent=2)

            print(f"✅ Language model successfully exported to {export_path}")
            print(f"✅ Tokenizer info saved to {tokenizer_path}")
            
            return export_path

        except Exception as e:
            print(f"❌ Language model export failed: {e}")
            print("💡 For mobile deployment, consider using a smaller model or rule-based captioning")
            return None

    def export_all(self):
        """
        Exports all models and generates a summary.
        """
        print("🚀 Starting model export process...")
        results = {}
        
        # Export YOLOv8
        yolo_path = self.export_yolo_model()
        results['yolo'] = yolo_path
        
        
        # Export language model (optional for mobile)
        print("\n⚠️ Note: Language model export for mobile is experimental")
        gemma_path = self.export_gemma_model()
        results['gemma'] = gemma_path
        
        # Create summary file
        summary_path = os.path.join(self.export_dir, 'models_summary.json')
        summary = {
            'yolo_model_path': yolo_path,
            'gemma_model_path': gemma_path,
            'yolo_classes_path': os.path.join(self.yolo_dir, 'class_names.json') if yolo_path else None,
            'tokenizer_info_path': os.path.join(self.gemma_dir, 'tokenizer_info.json') if gemma_path else None,
            'yolo_export_successful': yolo_path is not None,
            'gemma_export_successful': gemma_path is not None,
            'recommendation': "Use YOLO for detection. Consider rule-based captioning if language model export failed."
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Export Summary saved: {summary_path}")
        
        if yolo_path:
            print("✅ YOLOv8 export: SUCCESS - Ready for Android deployment")
        else:
            print("❌ YOLOv8 export: FAILED")
        
        if gemma_path:
            print("✅ Language model export: SUCCESS - Experimental mobile support")
        else:
            print("⚠️ Language model export: FAILED - Recommend rule-based captioning for mobile")
        
        print(f"\n📁 Models saved in: {self.export_dir}")
        print("\n💡 Next steps:")
        print("1. Copy the .ptl files to your Android app's assets folder")
        print("2. Use the class_names.json for object detection labels")
        print("3. Implement rule-based captioning as fallback")


if __name__ == "__main__":
    # Example usage
    saver = ModelSaver()
    saver.export_all()