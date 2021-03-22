## image processing shit
import math
from typing import Tuple, Union, List
from itertools import combinations, product

import cv2
# import easyocr
import pytesseract
import numpy as np
from PIL import Image
# import face_recognition
from deskew import determine_skew

## initialize reader
# def init_easyocr():
    # global reader
    # reader = easyocr.Reader(['en'], gpu = False)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

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
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue = background)

def get_image_variations(images: list) -> list:

    all_images = []
    for image in images:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharp_gray = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,17,15)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,35,30)

        all_images.extend([image, gray, sharp_gray, thresh1, thresh2])

    return all_images

# def is_image_upside_down(img: np.array) -> bool:
    
#     face_locations = face_recognition.face_locations(img)
#     encodings = face_recognition.face_encodings(img, face_locations)
#     image_is_upside_down = (len(encodings) == 0)
#     return image_is_upside_down

def fix_orientation(image: np.array) -> np.array:

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    # if is_image_upside_down(image):
        # angle += 180
    
    rotated = rotate(image, angle, (0, 0, 0)) 
    alt_rotated = rotate(image, angle+180, (0, 0, 0))
    return [rotated, alt_rotated]

# def read_text_easyocr(image: np.array, threshold = 0.5) -> list:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     result = reader.readtext(gray)
#     result = sorted(result, key = lambda x: x[2], reverse = True)
#     result = list(filter(lambda x: x[2]>threshold, result))
#     result = list(map(lambda x: x[1], result))
#     return result

def read_text_pytess(image: np.array, min_length = 2) -> list:
    result = pytesseract.image_to_string(image)
    result = result.split('\n')
    result = list(filter(lambda x: len(x) >= min_length, result))
    return result

### perspective correction
is_vertical = lambda theta: theta > 2.356 or theta < 0.7854
is_horizontal = lambda theta: theta <= 2.356 and theta >= 0.7854

def get_line_coordinates(rho, theta, x_offset = 0, y_offset = 0):
    
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho + x_offset
    y0 = b*rho + y_offset
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    return x1, y1, x2, y2

def get_bounding_lines(image, min_length_ratio = 0.3, pad_ratio = 0.2):
    
    height, width, _ = image.shape
    min_height = int(height*min_length_ratio)
    min_width = int(width*min_length_ratio)
    
    x_offset = int(width*pad_ratio)
    y_offset = int(height*pad_ratio)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    
    hlines = cv2.HoughLines(edges, 1, np.pi/180, min_width)
    if hlines is None:
        hlines = []
    else:
        hlines = filter(lambda line: is_horizontal(line[0][1]), hlines)
        hlines = map(lambda line: get_line_coordinates(line[0][0], line[0][1], x_offset, y_offset), hlines)
    
    vlines = cv2.HoughLines(edges, 1, np.pi/180, min_height)
    if vlines is None:
        vlines = []
    else:
        vlines = filter(lambda line: is_vertical(line[0][1]), vlines)
        vlines = map(lambda line: get_line_coordinates(line[0][0], line[0][1], x_offset, y_offset), vlines)
    
    return list(hlines), list(vlines)

def pad_image(image, pad_ratio = 0.2):
    
    height, width, _ = image.shape
    x_offset = int(width*pad_ratio)
    y_offset = int(height*pad_ratio)
    
    nimage = cv2.copyMakeBorder( image, y_offset, y_offset, x_offset, x_offset, cv2.BORDER_CONSTANT)
    return nimage

def get_intersection_point(hline, vline):

    x1, y1, x2, y2 = hline
    x3, y3, x4, y4 = vline

    d = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    px = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
    px = int(px/d)

    py = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
    py = int(py/d)
    
    return px, py

def get_polygon_area(corners):
    n = len(corners) 
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def get_largest_box(hlines, vlines, min_area):
    hpairs = combinations(hlines, 2)
    vpairs = combinations(vlines, 2)

    boxes = product(hpairs, vpairs)
    largest_box_area = min_area
    largest_box = None

    for box in boxes:
        (l1, l2), (l3, l4) = box
        a = get_intersection_point(l1, l3)
        b = get_intersection_point(l1, l4)
        d = get_intersection_point(l2, l3)
        c = get_intersection_point(l2, l4)

        area = get_polygon_area([a, b, c, d])

        if area > largest_box_area:
            largest_box = [a, b, c, d]
            largest_box_area = area

    if largest_box is None:
        return None

    largest_box = order_points(np.array(largest_box))
    return largest_box

def order_points(pts):
    # the first entry in the list is the top-left, 
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    
    return rect

def correct_perspective(image, min_length_ratio = 0.3, pad_ratio = 0.2, min_area_ratio = 0.2, out_width = 500, out_height = 400):

    hlines, vlines = get_bounding_lines(image, min_length_ratio = min_length_ratio, pad_ratio = pad_ratio)
    if len(hlines) == 0 or len(vlines) == 0:
        return image

    min_area = image.shape[0]*image.shape[1]*min_area_ratio

    orig_pts = get_largest_box(hlines, vlines, min_area=min_area)
    if orig_pts is None:
        return image

    # Coordinates that you want to Perspective Transform
    new_pts = np.float32([[0, 0], [out_width, 0], [0, out_height], [out_width, out_height]])
    ptransform = cv2.getPerspectiveTransform(orig_pts, new_pts)

    image = pad_image(image, pad_ratio = pad_ratio)
    image = cv2.warpPerspective(image, ptransform, (out_width, out_height))
    return image

def read_text_process(image: np.array, method = 'pytess') -> list:

    image = correct_perspective(image, out_width = 500, out_height = 400)
    images = fix_orientation(image)
    images = get_image_variations(images)

    text_blobs = []
    for image in images:
        if method  == 'pytess':
            text_blobs.extend(read_text_pytess(image))

    text_blobs = list(map(lambda x: x.lower(), text_blobs))
    return text_blobs