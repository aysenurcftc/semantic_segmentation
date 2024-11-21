import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, LEARNING_RATE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, \
    VAL_MASK_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, LOAD_MODEL, NUM_EPOCHS, CHECKPOINT_PATH, SAVE_MODEL_PATH, \
    MODEL_NAME
from models.attention_unet import AttentionUnet
from models.nestedUnet import NestedUNet
from models.unet import UNET
from utils.transforms import get_train_transforms, get_val_transforms
from utils.utils import save_checkpoint, check_accuracy, save_predictions_as_imgs, load_checkpoint, get_loaders


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = get_train_transforms()

    val_transforms = get_val_transforms()

    if MODEL_NAME == "nested_unet":
        model = NestedUNet(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL_NAME == "attention_unet":
        model = AttentionUnet(in_channels=3, out_channels=1).to(DEVICE)
    else:
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, SAVE_MODEL_PATH)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
   main()