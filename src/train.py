import os.path

import cv2
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from src.dataset import CurveDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class KeyPointModel(nn.Module):
    def __init__(self, number_key_points=10):  # Assuming 10 key points (x, y) pairs
        super(KeyPointModel, self).__init__()
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.5,  # dropout ratio, default is None
            activation='sigmoid',  # activation function, default is None
            classes=4,  # define number of output labels
        )
        self.features = smp.Unet(
            encoder_name="resnet152",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            aux_params=aux_params
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 416, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, number_key_points * 2)
        )

    def forward(self, x):
        mask, label = self.features(x)
        x = mask.view(-1, 256 * 416)
        x = self.classifier(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def curve_loss(predicted, target):
    # Reshape to (num_keypoints, 2)
    predicted = predicted.view(predicted.size(0), -1, 2)
    target = target.view(target.size(0), -1, 2)

    # Calculate MSE loss
    loss = F.mse_loss(predicted, target)

    return loss.float()


def train_loop(model, dataloader, loss_function, optimizer, desc):
    model.train()
    with trange(len(dataloader.dataset), desc=desc) as t:
        running_loss = 0.0
        batch_count = 0
        for images, key_points in dataloader:
            batch_count += 1
            images = images.to(device)
            key_points = key_points.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, key_points)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            t.set_postfix(loss=f'{loss.item():.6f}', avg_loss=f'{(running_loss / batch_count):.6f}')
            t.update(len(images))
    return running_loss / len(dataloader)


def evaluate_image(input_path, output_path):
    model.eval()
    image = cv2.imread(input_path)
    image = cv2.resize(image, (416, 256))
    data = image.transpose(2, 0, 1)
    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
    output = model(data)
    output = output.view(-1, 2).detach().cpu().numpy()
    # cv2.drawKeypoints(image, [cv2.KeyPoint(x / 100 * image.shape[1], y / 100 * image.shape[0], 1) for x, y in output], image, color=(0, 255, 0))
    cv2.drawKeypoints(image, [cv2.KeyPoint(x, y, 1) for x, y in output], image, color=(0, 255, 0))
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    dataset = CurveDataset(number_key_points=10)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = KeyPointModel(number_key_points=10)
    model = model.to(device)
    if os.path.exists('model_weights.pth'):
        try:
            model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device)))
            print("load pretrain weight")
        except Exception as e:
            print(f"fail to load weight: {e}")
    model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = curve_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5)

    num_epochs = 1000
    for epoch in range(num_epochs):
        loss = train_loop(model, train_loader, loss_function, optimizer, f'Epoch {epoch + 1}/{num_epochs}')
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(loss)
        torch.save(model.state_dict(), 'model_weights.pth')

    evaluate_image("../data/vaptcha-recover/images/1b5c9d60-348fa791ac9f412c978630bd040e6c7f.jpg", "1.png")
    evaluate_image("test.png", "2.png")
