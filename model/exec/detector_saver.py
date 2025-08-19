import os,sys
parent_dir = os.path.abspath(os.path.join(".."))
if not parent_dir in sys.path:
    sys.path.append(parent_dir)
from main.detector import YOLOConverter
from torch.utils.mobile_optimizer import optimize_for_mobile
model_path='yolov8n.pt'
save_path='../saved/detector/yolov8n_mobile.ptl'
class_names_path = '../saved/detector/detector_class_names.txt'

if __name__ == "__main__":
    opt = YOLOConverter(model_path)
    opt.export_to_ptl(save_path=save_path)
    opt.save_class_names(class_names_path)