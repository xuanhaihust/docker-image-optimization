import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from app.src.serving import nvidia_serving_factory as serving


def classify_document(image, model_name='classify-document-default'):
    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.ToTensor()
        ]
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pillow = Image.fromarray(image)

    img_transformed = transform(image_pillow)
    img_process = np.expand_dims(img_transformed.numpy(), axis=0)

    prob = serving().classify(model_name, img_process)

    return np.argmax(prob[0])
