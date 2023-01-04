import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from app.src.serving import nvidia_serving_factory as serving


def classify_resnet(image, model_name, ROI, classes=None):
    """
    Classify type of images
    """
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ROI_x0, ROI_y0, ROI_x1, ROI_y1 = float(dict(ROI[0])['x']), \
                                     float(dict(ROI[0])['y']), \
                                     float(dict(ROI[1])['x']), \
                                     float(dict(ROI[1])['y'])

    h, w, c = image.shape
    image_x0 = int(ROI_x0 * w)
    image_y0 = int(ROI_y0 * h)
    image_x1 = int(ROI_x1 * w)
    image_y1 = int(ROI_y1 * h)
    image_ROI = image[image_y0:image_y1, image_x0:image_x1, :]

    augmented = transform(image=image_ROI)
    image_ROI = augmented['image']
    image_process = image_ROI.numpy()
    image_process = np.expand_dims(image_process, axis=0)

    prob = serving().classify(model_name, image_process)

    label = np.argmax(prob[0])
    softmax = torch.nn.Softmax(dim=0)
    probability = softmax(torch.Tensor(prob[0])).numpy()

    if classes is not None:
        return classes[label], np.max(probability)

    return label, np.max(probability)
