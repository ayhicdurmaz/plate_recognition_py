import cv2
import imutils
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import os

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Output klasörü oluştur
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load image
img = cv2.imread("I0.png")
if img is None:
    raise FileNotFoundError("Cannot load image 'I0.png'. Check the path.")
cv2.imwrite(os.path.join(output_dir, "01_original.png"), img)
orig = img.copy()

# Convert to grayscale
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, "02_gray.png"), gray)

# Blackhat morphology
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
cv2.imwrite(os.path.join(output_dir, "03_blackhat.png"), blackhat)

# Light regions
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
_, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(output_dir, "04_light.png"), light)

# Gradient in X direction
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
if maxVal - minVal != 0:
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
cv2.imwrite(os.path.join(output_dir, "05_gradX_raw.png"), gradX)

# Smooth and close gaps
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
_, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(output_dir, "06_thresh.png"), thresh)

# Erode and dilate
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# Mask with light regions
thresh = cv2.bitwise_and(thresh, thresh, mask=light)
thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)
cv2.imwrite(os.path.join(output_dir, "07_thresh_masked.png"), thresh)

# Find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

plate_image = []
for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if 3.0 < ar < 5.0:
        plate = gray[y:y+h, x:x+w]
        plate_image.append(plate)
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"08_plate_candidate_{i}.png"), plate)

cv2.imwrite(os.path.join(output_dir, "09_detected_plates_on_orig.png"), orig)

reader = easyocr.Reader(['en'], gpu=False)
for i, im in enumerate(plate_image):

    angles = np.arange(-5, 5.5, 0.5)
    best_angle = 0
    max_variance = 0

    for angle in angles:
        rotated = rotate_image(im, angle)
        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)
        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    im = rotate_image(im, best_angle)
    cv2.imwrite(os.path.join(output_dir, f"10_plate_{i}_rotated.png"), im)

    h, w = im.shape
    new_w = 400
    scale = new_w / w
    new_h = int(h * scale)
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, f"11_plate_{i}_resized.png"), resized)

    denoised = cv2.medianBlur(resized, 5)
    cv2.imwrite(os.path.join(output_dir, f"12_plate_{i}_denoised.png"), denoised)

    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, None, iterations=2)
    cv2.imwrite(os.path.join(output_dir, f"13_plate_{i}_binary.png"), binary)

    texts = reader.readtext(binary, detail=0)
    full_text = ' '.join(texts)
    print(f"Detected text for plate {i}: {full_text}")
