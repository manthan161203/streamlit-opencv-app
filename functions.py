import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def convert_to_grayscale(image):
    """Convert image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_canny_edge(image, low_threshold, high_threshold):
    """Apply Canny edge detection to the image"""
    gray = convert_to_grayscale(image)
    return cv2.Canny(gray, low_threshold, high_threshold)


def adjust_brightness(image, factor):
    """Adjust image brightness"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(img_pil)
    enhanced_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)


def adjust_contrast(image, factor):
    """Adjust image contrast"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(img_pil)
    enhanced_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)


def apply_blur(image, blur_type, kernel_size):
    """Apply different types of blur to the image"""
    if blur_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif blur_type == "Median Blur":
        return cv2.medianBlur(image, kernel_size)
    elif blur_type == "Box Blur":
        return cv2.blur(image, (kernel_size, kernel_size))
    return image


def apply_threshold(image, threshold_value, max_value, threshold_type):
    """Apply thresholding to the image"""
    gray = convert_to_grayscale(image)
    if threshold_type == "Binary":
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    elif threshold_type == "Binary Inverted":
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    elif threshold_type == "Truncate":
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TRUNC)
    elif threshold_type == "To Zero":
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO)
    elif threshold_type == "To Zero Inverted":
        _, thresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO_INV)
    elif threshold_type == "Adaptive Mean":
        thresh = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == "Adaptive Gaussian":
        thresh = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == "Otsu's Binarization":
        _, thresh = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        return gray
    
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def apply_color_filter(image, filter_type):
    """Apply different color filters to the image"""
    if filter_type == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(image, kernel)
    elif filter_type == "Negative":
        return cv2.bitwise_not(image)
    elif filter_type == "Blue Boost":
        b, g, r = cv2.split(image)
        b = cv2.convertScaleAbs(b, alpha=1.2, beta=0)
        return cv2.merge([b, g, r])
    elif filter_type == "Red Boost":
        b, g, r = cv2.split(image)
        r = cv2.convertScaleAbs(r, alpha=1.2, beta=0)
        return cv2.merge([b, g, r])
    elif filter_type == "Green Boost":
        b, g, r = cv2.split(image)
        g = cv2.convertScaleAbs(g, alpha=1.2, beta=0)
        return cv2.merge([b, g, r])
    return image


def detect_faces(image):
    """Detect faces in the image and draw rectangles around them"""
    gray = convert_to_grayscale(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    
    img_copy = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return img_copy, len(faces)


def resize_image(image, width, height):
    """Resize image to specified dimensions"""
    return cv2.resize(image, (width, height))


def crop_image(image, x_start, y_start, x_end, y_end):
    """Crop image to specified coordinates"""
    return image[y_start:y_end, x_start:x_end]


def rotate_image(image, angle):
    """Rotate image by specified angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def flip_image(image, flip_code):
    """Flip image horizontally or vertically
    flip_code: 0 for vertical flip, 1 for horizontal flip
    """
    return cv2.flip(image, flip_code)


def apply_morphological_op(image, op_type, kernel_size):
    """Apply morphological operations"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = convert_to_grayscale(image)
    
    if op_type == "Erosion":
        result = cv2.erode(gray, kernel, iterations=1)
    elif op_type == "Dilation":
        result = cv2.dilate(gray, kernel, iterations=1)
    elif op_type == "Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif op_type == "Closing":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif op_type == "Gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    else:
        return image
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def detect_contours(image, threshold1=100, threshold2=200):
    """Detect and draw contours in the image"""
    gray = convert_to_grayscale(image)
    edges = cv2.Canny(gray, threshold1, threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_copy = image.copy()
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 2)
    return img_copy