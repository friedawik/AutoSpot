from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from IPython import embed

# Load a pretrained YOLO11n model
# model = YOLO("yolo11n-seg.pt")
model = YOLO("runs/segment/train3/weights/best.pt")
# img = '../../data/patch_256/test/images/MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x0_y1.tif'
# img = '../../data/patch_256_8bit/test/images/MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x0_y1.png'
# Run inference on 'bus.jpg' with arguments
# results = model.predict(img, save=True, imgsz=256, conf=0.3)
# metrics = model.val(data="../../data/patch_256_8bit/dataset.yaml", save_json=True)
# embed()
metrics = model.val(data="../../data/patch_256_8bit/dataset.yaml",split="test", plots=True)
embed()


# Get segmentation performance
print(metrics.results_dict)



