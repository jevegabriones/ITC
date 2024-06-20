# -*- coding: utf-8 -*-


# Getting the installation ready:
# Install detectron2
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Check the version of pytorch and cuda
import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Prepare the dataset:
# Register the training and validation datasets in COCO format
from detectron2.data.datasets import register_coco_instances

# Define paths
train_json_path_C = "/scratch/depfg/vegab002/ITC/Training_Dataset/Canada_Dataset/annotations/Quebec_train.json"
train_images_path_C = "/scratch/depfg/vegab002/ITC/Training_Dataset/Canada_Dataset/images"

train_json_path_M = "/scratch/depfg/vegab002/ITC/Training_Dataset/Malaysia_Dataset/annotations/Malaysia.json"
train_images_path_M = "/scratch/depfg/vegab002/ITC/Training_Dataset/Malaysia_Dataset/images"

val_json_path = "/scratch/depfg/vegab002/ITC/Validation_Dataset/annotations/Validation.json"
val_images_path = "/scratch/depfg/vegab002/ITC/Validation_Dataset/images"

# Register the datasets
register_coco_instances("my_dataset_train_C", {}, train_json_path_C, train_images_path_C)
register_coco_instances("my_dataset_train_M", {}, train_json_path_M, train_images_path_M)
register_coco_instances("my_dataset_val", {}, val_json_path, val_images_path)

# Visualize the training datasets:
# # Gather metadata for visualizing the samples
# train_metadata_C = MetadataCatalog.get("my_dataset_train_C")
# train_dataset_dicts_C = DatasetCatalog.get("my_dataset_train_C")

# train_metadata_M = MetadataCatalog.get("my_dataset_train_M")
# train_dataset_dicts_M = DatasetCatalog.get("my_dataset_train_M")

# # Visualize random samples
# for d in random.sample(train_dataset_dicts_C, 1):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata_C, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2_imshow(out.get_image()[:, :, ::-1])

# Define and train the model:
# Set the hyperparameters
from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train_C", "my_dataset_train_M",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # number of classes (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# Create output directory and train the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Look at training curves in tensorboard:
# tensorboard --logdir output

# Inference & evaluation using the trained model:
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Obtain metadata for the validation dataset
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

# Import necessary libraries
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode

# Initialize an empty list to store all predictions
all_predictions = []

# Loop over the validation dataset
for idx, d in enumerate(val_dataset_dicts):

    # Get the image and its predictions
    # Format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  
    
    # Use Visualizer to draw the predictions on the image.
    v1 = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    v2 = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)

    # Draw and convert images
    img1 = v1.draw_dataset_dict(d).get_image()[:, :, ::-1]
    img2 = v2.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

    # Plot images side by side
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(img1); ax[0].axis('off'); ax[0].set_title('True Masks')
    ax[1].imshow(img2); ax[1].axis('off'); ax[1].set_title('Predicted Masks')

    # Save the figure with a unique name
    fig_name = f"/scratch/depfg/vegab002/ITC/output/figure_{idx}.png"
    plt.savefig(fig_name)

    # Convert the mask to COCO format and save it
    coco_format = {
        "image_id": idx,
        "category_id": 1,  # Assuming there's only one category
        "bbox": BoxMode.convert(outputs["instances"].pred_boxes.tensor.tolist(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS),
        "segmentation": outputs["instances"].pred_masks.tolist(),  # Binary mask to RLE, might need conversion depending on your setup
        "score": outputs["instances"].scores.tolist(),
    }
    all_predictions.append(coco_format)

# Save all predictions in a single json file
with open("/scratch/depfg/vegab002/ITC/output/validation_predicted.json", "w") as f:
    json.dump(all_predictions, f)

# Evaluate its performance using AP metric implemented in COCO API
# Another equivalent way to evaluate the model is to use `trainer.test`
# For reference, the balloon dataset provided in detectron2 gives an AP of ~70
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
