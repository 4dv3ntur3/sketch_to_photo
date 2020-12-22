import cv2
import numpy as np

# Kmeans color segmentation
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

# Load image and perform kmeans
image = cv2.imread('./n02958343_12624.jpg')
original = image.copy()
kmeans = kmeans_color_quantization(image, clusters=4)

# Convert to grayscale, Gaussian blur, adaptive threshold
gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

# Draw largest enclosing circle onto a mask
mask = np.zeros(original.shape[:2], dtype=np.uint8)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    ((x, y), r) = cv2.minEnclosingCircle(c) #여길 손대야 할듯 
    cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    break

# Bitwise-and for result
result = cv2.bitwise_and(original, original, mask=mask)
result[mask==0] = (255,255,255)

cv2.imshow('thresh', thresh)
cv2.imshow('result', result)
cv2.imshow('mask', mask)
cv2.imshow('kmeans', kmeans)
cv2.imshow('image', image)
cv2.waitKey()