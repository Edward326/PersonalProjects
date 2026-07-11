import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.classifier import YOLOClassifierConverter
model_path='../saved/original/classifier/yolov8n-cls.pt'
save_path='../saved/mobile/classifier/yolov8n_cls_mobile.onnx'
class_names_path = '../saved/mobile/classifier/classifier_class_names.txt'

if __name__ == "__main__":
    opt = YOLOClassifierConverter(model_path)
    opt.export_to_onnx_simple(save_path)
    opt.save_class_names(class_names_path)