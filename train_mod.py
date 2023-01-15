"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import NUM_PTS, CROP_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

import pretrainedmodels
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def lr(step):
    return 0.9 ** step


def main(args):
    os.makedirs("runs", exist_ok=True)


    albumentations_transform_pos = A.Compose([
                A.ShiftScaleRotate(p=0.01),
                A.Rotate([-25, 25], p=0.01)
    ],  keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


    albumentations_transform = A.Compose([
                A.Blur(p=0.02),
                A.ColorJitter(p=0.02),
                A.RandomBrightnessContrast(p=0.02),
                A.Sharpen(p=0.02),
                A.CoarseDropout(min_holes=2, max_holes=8, min_width=5, max_width=20, min_height=5, max_height=20, p=0.01),
                A.ImageCompression(p=0.01),
                A.Downscale(p=0.01),
                A.ToGray(p=0.01)
    ])


    augmentation_transform = transforms.Compose([
            TransformByKeys(lambda sample: albumentations_transform(image=np.array(sample["image"])), ('image', ), album=True),
            TransformByKeys(lambda sample: albumentations_transform_pos(image=np.array(sample["image"]), keypoints=np.array(sample["landmarks"]).reshape(-1, 2)), ('image', 'landmarks'), album=True),
            ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
            CropCenter(CROP_SIZE),
            TransformByKeys(transforms.ToTensor(), ("image", )),
            TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",))
   ])


    test_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToTensor(), ("image", )),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image", ))
    ])


    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), augmentation_transform, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), test_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    model = models.densenet161(pretrained=True)
    model.requires_grad_(True)


    #model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Linear(model.classifier.in_features, 2 * NUM_PTS, bias=True)

    ##### Fine tuning  ##################################################################
    # with open(os.path.join("runs", "resnet50@224_best.pth"), "rb") as fp:
    #     best_state_dict = torch.load(fp, map_location="cpu")
    #     model.load_state_dict(best_state_dict)


    # #model.layer2.requires_grad_(True)
    # model.layer3.requires_grad_(True)
    # model.layer4.requires_grad_(True)
    #####################################################################################

    #model.avg_pool.requires_grad_(True)    
    model.classifier.requires_grad_(True)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    loss_fn = fnn.mse_loss

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, scheduler, device=device)
        val_loss = validate(model, val_dataloader, loss_fn, device=device)
        
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), test_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
