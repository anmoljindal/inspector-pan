## image processing shit
import math
from typing import Tuple, Union, List

import cv2
import easyocr
import numpy as np
from PIL import Image
import face_recognition
from deskew import determine_skew

## initialize reader
reader = easyocr.Reader(['en'], gpu=False)

def read_image(image) -> np.array:
    try:
        image = Image.open(image).convert('RGB') 
        image = np.array(image) 
        image = image[:, :, ::-1].copy() # Convert RGB to BGR 
        return image
    except:
        return

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def is_image_upside_down(img: np.array) -> bool:
    
    face_locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, face_locations)
    image_is_upside_down = (len(encodings) == 0)
    return image_is_upside_down

def fix_orientation(image: np.array) -> np.array:

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if is_image_upside_down(image):
        angle += 180
    
    rotated = rotate(image, angle, (0, 0, 0)) 
    return rotated

def read_text(image: np.array, threshold=0.5) -> list:
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    result = sorted(result, key=lambda x: x[2], reverse=True)
    result = list(filter(lambda x: x[2]>threshold, result))
    return result

def read_text_process(image: np.array) -> list:

    text_blobs = read_text(image)
    text_blobs = list(map(lambda x: x[1].lower(), text_blobs))
    return text_blobs