import os
import uuid
from typing import BinaryIO

import numpy as np
import re
import requests
from requests.exceptions import HTTPError

import cv2
import time
import logging
import base64
from PIL import Image


URL_PATTERN = r"^https?:\/\/.+"


def bytes_to_cv2(img_bytes):
    data = np.frombuffer(img_bytes, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def base64_to_cv2(b64str):
    img_bytes = base64.b64decode(b64str.encode('utf8'))
    img_np = bytes_to_cv2(img_bytes)
    return img_np


def pillow_to_cv2(pil_image: Image):
    # pil_image = pil_image.convert('RGB')  # in case of gray scale input
    img_np = np.array(pil_image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2


def cv2_to_file(img_cv2, img_path=None):
    """convert cv2 numpy array to a file in case io.BytesIO can't be used"""
    if not img_path:
        img_path = uuid.uuid4().hex + '.jpg'

    cv2.imwrite(img_path, img_cv2)
    img_file = open(img_path, 'rb')
    os.remove(img_path)

    return img_file


def bytes_to_file(img_bytes: bytes, img_path: str = None) -> BinaryIO:
    """convert bytes object to a file in case io.BytesIO can't be used"""
    if not img_path:
        img_path = uuid.uuid4().hex + '.jpg'

    # write file to
    with open(img_path, 'wb') as f:
        f.write(img_bytes)
    img_file = open(img_path, 'rb')
    os.remove(img_path)

    return img_file


def read_image_from_url(img_url: str):
    """
    Read img from an url.
    The img_url can be a https internet url or path to a local file
    """
    try:
        start = time.time()
        if re.match(URL_PATTERN, img_url):
            requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':RC4-SHA'
            r = requests.get(img_url, stream=True)
            r.raise_for_status()
            buff = r.content
        else:
            # url is local file for testing
            with open(img_url, "rb") as f:
                buff = f.read()
        img = np.frombuffer(buff, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        end = time.time()
        logging.info("Read image %s in %0.3fms" % (img_url, (end - start) * 1000))
        return img
    except HTTPError:
        raise HTTPError(f"Cannot read image from url {img_url}")
    except Exception as err:
        raise Exception(f"Cannot read image from url {img_url}, error: {err}")


def resize_one_edge(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize keep image ratio.
    :param image: HxWxC
    :param width: image width
    :param height: image height
    :param inter: cv2 interpolation algorithm
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_max_edge(img: np.ndarray, max_edge=1024) -> np.ndarray:
    H, W, _ = img.shape
    if H > W:
        img = resize_one_edge(img, height=max_edge)
    else:
        img = resize_one_edge(img, width=max_edge)
    return img


def resize_to_size(img: np.ndarray, dst_height, dst_width):
    height, width = img.shape[:2]
    if dst_width * dst_height < width * height:  # shrink
        img = cv2.resize(img, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
    else:  # enlarge
        img = cv2.resize(img, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
    return img


def encode_image_for_cache(img_np, encode_format='jpg'):
    _, img_out = cv2.imencode('.' + encode_format, img_np)

    return img_out
