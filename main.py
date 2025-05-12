import cv2
import imutils
import numpy as np
import easyocr
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Load image
img = cv2.imread("I0.png")
if img is None:
    raise FileNotFoundError("Cannot load image 'I0.png'. Check the path.")
orig = img.copy()

# Convert to grayscale
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

#median = cv2.medianBlur(gray, 5)

# Apply blackhat morphology to find dark regions on light background
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

# Find light regions for masking
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
_, light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Compute gradient in X direction and scale to 0-255
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
if maxVal - minVal != 0:
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")

# Smooth and close gaps
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
_, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Erode and dilate to remove small blobs
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# Mask with light regions
thresh = cv2.bitwise_and(thresh, thresh, mask=light)
thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)

# Find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

plate_image = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    # Typical license plate aspect ratio filter
    if 3.0 < ar < 5.0:
        # Extract plate region
        plate = gray[y:y+h, x:x+w]
        plate_image.append(plate)
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)


reader = easyocr.Reader(['en'], gpu=False)
# Show individual plates
for i, im in enumerate(plate_image):

    angles = np.arange(-5, 5.5, 0.5)  # -5° ile +5° arası 0.5° adımlarla dene
    best_angle = 0
    max_variance = 0

    for angle in angles:
        rotated = rotate_image(im, angle)  # senin rotate_image fonksiyonun
        projection = np.sum(rotated, axis=1)  # satır bazlı projeksiyon
        variance = np.var(projection)  # varyans: harf + boşluk kontrastı
        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    im = rotate_image(im, best_angle)

    h, w = im.shape
    new_w = 400
    scale = new_w / w
    new_h = int(h * scale)

    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    denoised = cv2.medianBlur(resized, 5)

    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.dilate(binary, None, iterations=2)

    texts = reader.readtext(binary, detail=0)
    full_text = ' '.join(texts)
    print("Detected text:", full_text)


    plt.figure(figsize=(10, 6))
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.show()

# Show plates on original image
plt.figure(figsize=(10,6))
plt.imshow(orig, cmap='gray')
plt.axis('off')
plt.show()

'''
plt.figure(figsize=(10,6))
plt.imshow(blackhat, cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,6))
plt.imshow(light, cmap='gray')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,6))
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()
'''