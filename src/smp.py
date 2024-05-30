import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# aux_params = dict(
#     pooling='avg',  # one of 'avg', 'max'
#     dropout=0.5,  # dropout ratio, default is None
#     activation='sigmoid',  # activation function, default is None
#     classes=4,  # define number of output labels
# )

model = smp.Unet(
    encoder_name="resnet152",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
    # aux_params=aux_params
)

model.to(device)

model.eval()


image = cv2.imread("test.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.resize(image, (416, 256))
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
ndarray = np.transpose(image, (2, 0, 1))
tensor = torch.from_numpy(ndarray).unsqueeze(0).float().to(device)
print(f"Shape of tensor: {tensor.shape}")

# masks, labels = model.predict(tensor)
masks = model.predict(tensor)
print(f"Shape of masks: {masks.shape}")

mask_pred = torch.sigmoid(masks).cpu()
mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
mask_pred /= mask_pred.max()

mask = (mask_pred * 255).astype(np.uint8)
# mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# cv2.imshow('Mask', mask_3ch)
# masked_image = cv2.bitwise_and(image, mask_3ch)
cv2.imshow('Masked Image', mask[0][0])

cv2.waitKey(0)
cv2.destroyAllWindows()
