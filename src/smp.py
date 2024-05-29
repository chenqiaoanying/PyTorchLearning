import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

aux_params = dict(
    pooling='avg',  # one of 'avg', 'max'
    dropout=0.5,  # dropout ratio, default is None
    activation='sigmoid',  # activation function, default is None
    classes=4,  # define number of output labels
)

model = smp.Unet(
    encoder_name="resnet152",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=3,  # model output channels (number of classes in your dataset)
    aux_params=aux_params
)

model.to(device)

model.eval()


image = cv2.imread("test.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.resize(image, (416, 256))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = np.transpose(image, (2, 0, 1))
print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)
image = torch.from_numpy(image).float()
print(image.shape)

with torch.no_grad():  # no need to calculate gradients during evaluation
    image = image.to(device)
    segmented_image, x = model(image)

# Convert the output tensor to a numpy array
segmented_image = segmented_image.cpu().numpy()

# Perform an argmax operation to get the most probable class for each pixel
segmented_image = segmented_image[0][0]

# Normalize the segmented image to range 0-255
segmented_image = (segmented_image / np.max(segmented_image) * 255).astype(np.uint8)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()