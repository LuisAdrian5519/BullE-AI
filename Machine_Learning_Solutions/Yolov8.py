# Import libraries

from ultralytics import YOLO

# Select a classification model for feeder-level scoring
model = YOLO("yolov8n-cls.yaml")

# Train using the FBSI feeder dataset configured in Cap_config.yaml
model.train(data=r"C:...\FBSI", epochs=50, batch=32, device='cpu', workers=0)

# model.train(data="config.yaml", epochs=15, batch=32, device=0, fraction = 0.20, workers=0, 
#             lr0=0.01,
#             lrf=0.001,
#             momentum=0.98,
#             weight_decay=0.001,
#             warmup_epochs=3,
#             warmup_momentum=0.98,
#             degrees=0.0,
#             translate=0.0,
#             scale=0.0,
#             flipud=0.0,
#             fliplr=0.0,
#             shear=0.0,
#             perspective=0.0,
#             mosaic=0.0,
#             mixup=0.0,
#             hsv_h=0.05,
#             hsv_s=0.45,
#             hsv_v=0.45,
#             copy_paste=0.5)

# model.train(data="config.yaml", epochs=15, batch=32, device=0, fraction = 0.20, workers=0, 
#             lr0=0.01,
#             lrf=0.001,
#             momentum=0.98,
#             weight_decay=0.001,
#             warmup_epochs=3,
#             warmup_momentum=0.98,
#             degrees=0.0,
#             translate=0.0,
#             scale=0.0,
#             flipud=0.0,
#             fliplr=0.0,
#             shear=0.0,
#             perspective=0.0,
#             mosaic=0.0,
#             mixup=0.0,
#             hsv_h=0.1,
#             hsv_s=0.9,
#             hsv_v=0.9,
#             copy_paste=1.0)


