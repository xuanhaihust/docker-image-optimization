import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from app.src.serving import nvidia_serving_factory as serving


def classify_orientation(image, model_name='classify-orientation'):
    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transforms.ToTensor()
        ]
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_PIL = Image.fromarray(image)

    img_tranform = transform(image_PIL)
    img_process = np.expand_dims(img_tranform.numpy(), axis=0)

    prob = serving().classify(model_name, img_process)

    return np.argmax(prob[0])


def trasform_image_paddle(image, resize_size=256, crop_size=224, scale=1.0/255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image 
    img_h, img_w = img_rgb.shape[:2]
    percent = float(resize_size) / min(img_w, img_h)
    w_rs = int(round(img_w * percent))
    h_rs = int(round(img_h * percent))
    img_rs = cv2.resize(img_rgb, (w_rs, h_rs))

    # Crop image
    w_crop, h_crop = crop_size, crop_size
    img_rs_h, img_rs_w = img_rs.shape[:2]
    w_start = (img_rs_w - w_crop) // 2
    h_start = (img_rs_h - h_crop) // 2

    w_end = w_start + w_crop
    h_end = h_start + h_crop
    img_croped =  img_rs[h_start:h_end, w_start:w_end, :]

    # Normalize
    img_nor = (img_croped.astype('float32') * scale - mean) / std

    # toCHWimage
    img_transpose = np.transpose(img_nor, (2,0,1))
    img_exp = np.expand_dims(img_transpose, axis=0).astype(np.float32)

    return img_exp


def classify_orientation_paddle(image, model_name='classify-orientation-paddle'):
    # Transform image
    img_tranform = trasform_image_paddle(image)
    #Run Serving
    prob = serving().classify_paddle(model_name, img_tranform)
    # Tranform result
    result = np.array([prob[0][0], prob[0][3], prob[0][2], prob[0][1]])

    return np.argmax(result)
