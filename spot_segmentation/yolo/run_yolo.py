from ultralytics import YOLO
from IPython import embed
# Load a pretrained model
# model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.pt")
# model = YOLO("runs/segment/train2/weights/best.pt")
# embed()
# Train the model
results = model.train(data="../../data/patch_256_detection/dataset.yaml", 
                        batch = 4, 
                        epochs=5, 
                        lr0=0.05, 
                        imgsz=256,
                        mask_ratio=1,
                        cls=0.1
                        )

metrics = model.val()
# Get segmentation performance
print(metrics.results_dict)
embed()