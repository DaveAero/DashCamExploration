# laneDetection.py
# By: David Burke

import cv2
import numpy as np

lastLeftFitAvg = lastRightFitAvg = None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def regionOfInterest(image):
    polygons = np.array([[
        (910, 900),   # bottom-left
        (1680, 900),   # bottom-right
        (1360, 700),  # top-right
        (1200, 700)   # top-left
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def makeCoordinates(image, lineParameters):
    slope, intercept = lineParameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))

    if slope == 0 or not np.isfinite([slope, intercept]).all():
        return None

    x1 = (y1 - intercept) / slope
    x2 = (y2 - intercept) / slope

    w = image.shape[1]
    x1 = int(np.clip(x1, -w, 2*w))
    x2 = int(np.clip(x2, -w, 2*w))
    return np.array([x1, y1, x2, y2])

def avgSlopeIntercept(image, lines):
    global lastLeftFitAvg, lastRightFitAvg
    leftFit, rightFit = [], []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x2 - x1) < 2:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # guard: ignore near-horizontal / near-zero slopes (prevents crazy x)
        if abs(slope) < 0.1:
            continue

        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))

    # choose current averages if available, otherwise fall back to last
    leftParams  = np.average(leftFit, axis=0)  if len(leftFit)  else lastLeftFitAvg
    rightParams = np.average(rightFit, axis=0) if len(rightFit) else lastRightFitAvg

    # if we still don't have something valid, skip drawing this frame
    if leftParams is None or rightParams is None:
        return None

    leftLine  = makeCoordinates(image, leftParams)
    rightLine = makeCoordinates(image, rightParams)

    lastLeftFitAvg, lastRightFitAvg = leftParams, rightParams
    return np.array([leftLine, rightLine], dtype=np.int32)

def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # skip NaN/inf just in case
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 100), 12)

    return lineImage

cap = cv2.VideoCapture(r"data\GRME0192.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    cannyImage = canny(frame)

    croppedImage = regionOfInterest(cannyImage)

    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    avgLines = avgSlopeIntercept(frame, lines)

    lineImage = displayLines(frame, avgLines) if avgLines is not None else np.zeros_like(frame)

    comboImage = cv2.addWeighted(frame, 0.9, lineImage, 1, 1)

    cv2.imshow('Result', comboImage)
    cv2.imshow('cannyImage', cannyImage)
    cv2.imshow('RegionOfInterestImage', croppedImage)
    if cv2.waitKey(1) & 0xFF==27: # ESC KEY to close
        break

cap.release()
cv2.destroyAllWindows()

