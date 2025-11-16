import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.detector import YOLOConverter
model_path='../saved/original/detector/yolov8n.pt'
save_path='../saved/mobile/detector/yolov8n_mobile.onnx'
class_names_path = '../saved/mobile/detector/detector_class_names.txt'

if __name__ == "__main__":
    opt = YOLOConverter(model_path)
    opt.export_to_onnx_simple(save_path)
    opt.save_class_names(class_names_path)