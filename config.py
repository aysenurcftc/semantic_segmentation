import torch

MODEL_NAME = ""
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL_PATH = "checkpoints/my_checkpoint.pth.tar"
CHECKPOINT_PATH = "checkpoints/my_checkpoint.pth.tar"
# Image dimensions
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# Data directories
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"

