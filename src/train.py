import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet152_Weights

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
        self.backbone = models.resnet152(weights=[ResNet152_Weights.DEFAULT])
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, number_key_points * 2)  # 10 key points, each with x and y

    def forward(self, x):
        return self.backbone(x)


def curve_loss(predicted, target):
    # Reshape to (num_keypoints, 2)
    predicted = predicted.view(predicted.size(0), -1, 2)
    target = target.view(target.size(0), -1, 2)

    # Calculate MSE loss
    loss = F.mse_loss(predicted, target)

    return loss.float()


dataset = CurveDataset(number_key_points=20)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, )

# Initialize model, optimizer
model = KeyPointModel(number_key_points=20)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = curve_loss

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints in dataloader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, keypoints)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

model.eval()


# test test.png
# image = cv2.imread("../data/vaptcha-recover/images/1b5c9d60-348fa791ac9f412c978630bd040e6c7f.jpg")
def evaluate_image(image):
    data = image.transpose(2, 0, 1)
    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
    output = model(data)
    output = output.view(-1, 2).detach().cpu().numpy()
    # cv2.drawKeypoints(image, [cv2.KeyPoint(x / 100 * image.shape[1], y / 100 * image.shape[0], 1) for x, y in output], image, color=(0, 255, 0))
    cv2.drawKeypoints(image, [cv2.KeyPoint(x, y, 1) for x, y in output], image, color=(0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey(0)


evaluate_image(cv2.imread("../data/vaptcha-recover/images/1b5c9d60-348fa791ac9f412c978630bd040e6c7f.jpg"))
evaluate_image(cv2.imread("test.png"))