import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.captioner import BLIPVQAWrapper,export_blip_vqa_to_onnx,quantize_onnx_model

if __name__ == "__main__":
    export_blip_vqa_to_onnx()
    
    # Optionally quantize the models
    export_dir = "../saved/captioner"
    onnx_files = [f for f in os.listdir(export_dir) if f.endswith('.onnx')]
    
    for onnx_file in onnx_files:
        onnx_path = os.path.join(export_dir, onnx_file)
        quantized_path = os.path.join(export_dir, f"quantized_{onnx_file}")
        quantize_onnx_model(onnx_path, quantized_path)