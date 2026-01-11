# laneDetection.py
# By: David Burke

##################### Import Libaries #####################
import cv2
import numpy as np

##################### Tuneable varaiables #####################
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSS_K = (5, 5)

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESH = 100
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 5

MIN_DX = 2
MIN_ABS_SLOPE = 0.1

lastLeftFitAvg = None
lastRightFitAvg = None

lastLeftFitAvg = lastRightFitAvg = None

##################### Functions #####################
# Setting up Canny for edge detection
def canny(image):
    #First convert image to Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Then blur the image to get more consitent results
    blur = cv2.GaussianBlur(gray, GAUSS_K, 0)
    # return the canny image with all edges highlighted
    return cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

# Setting up region of interest to reduce computational workload
def regionOfInterest(image):
    # Through trial and error this was found the be the correct area
    polygons = np.array([[
        # bottom left
        (910, 900),
        # bottom right
        (1680, 900),
        # top right
        (1360, 700),
        # top left
        (1200, 700)
    ]], dtype=np.int32)

    # start with a black image with the same shape and format as the image
    mask = np.zeros_like(image)
    # apply the region of interest as a white area on this black image
    cv2.fillPoly(mask, polygons, 255)
    # Turns the region of interest from white back to the origional colour while it has kept everything outside of the region of interest as black
    return cv2.bitwise_and(image, mask)

# Take a line equation and turn it into two points on the screen
def makeCoordinates(image, lineParameters):
    # The import line equation
    slope, intercept = lineParameters
    h, w = image.shape[:2]

    # choose y-range to draw within
    y1 = h
    y2 = int(h * (3 / 5))

    # Do not proceed if slope is = 0 as this breaks intercept function below
    if slope == 0 or not np.isfinite([slope, intercept]).all():
        return None

    # Using y and the line equation to calculate x
    x1 = (y1 - intercept) / slope
    x2 = (y2 - intercept) / slope

    # clamp to sane range (still allows off-screen a bit)
    x1 = int(np.clip(x1, -w, 2 * w))
    x2 = int(np.clip(x2, -w, 2 * w))
    # Return the points to be drawn on screen and converts to intergers for openCV
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def avgSlopeIntercept(image, lines):
    # Using global variable which can be used across all frames of the video to help smooth the result
    global lastLeftFitAvg, lastRightFitAvg

    # If not lines are found, do not proceed
    if lines is None:
        return None

    # Setting up variables
    leftFit = []
    rightFit = []

    # For both lines
    for line in lines:
        # reshaping the Hough lines into 2 points
        x1, y1, x2, y2 = line.reshape(4)

        # A filter to remove vertical lines
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < MIN_DX:
            continue
        
        # A filter to remove horizontal lines
        slope = dy / dx
        if abs(slope) < MIN_ABS_SLOPE:
            continue

        # Compute the intercept point
        intercept = y1 - slope * x1

        # compute the lenght, longer lines are given greater weight
        length = float(np.hypot(dx, dy))

        # split into left or right line based on the slope 
        if slope < 0:
            leftFit.append((slope, intercept, length))
        else:
            rightFit.append((slope, intercept, length))

    # weighted average by length to find lines that fit
    def weighted_avg(fits, last):
        # if nothing fits, fall back to last point
        if not fits:
            return last
        # computing weighted average
        slopes = [s for s, b, L in fits]
        ints = [b for s, b, L in fits]
        wts = [L for s, b, L in fits]
        return np.array([
            np.average(slopes, weights=wts),
            np.average(ints, weights=wts)
        ])

    # run the weighted average update
    leftParams = weighted_avg(leftFit, lastLeftFitAvg)
    rightParams = weighted_avg(rightFit, lastRightFitAvg)

    # if no lines have been found, stop for this frame
    if leftParams is None and rightParams is None:
        return None

    # convert the average line equation into pixel coordinates
    leftLine = makeCoordinates(image, leftParams) if leftParams is not None else None
    rightLine = makeCoordinates(image, rightParams) if rightParams is not None else None

    # update the global last-known only when valid
    if leftParams is not None:
        lastLeftFitAvg = leftParams
    if rightParams is not None:
        lastRightFitAvg = rightParams

    out = []
    if leftLine is not None:
        out.append(leftLine)
    if rightLine is not None:
        out.append(rightLine)

    return np.array(out, dtype=np.int32) if out else None

# Display the found road marking on the origional image for viewer
def displayLines(image, lines):
    lineImage = np.zeros_like(image)

    if lines is None:
        return lineImage

    for x1, y1, x2, y2 in lines:
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        cv2.line(lineImage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 100), 12)

    return lineImage

# Draw the road center on the origional image to show if the car is centered or not.
def x_at_y(line, y):
    x1, y1, x2, y2 = line
    dy = (y2 - y1)
    if dy == 0:
        return None
    t = (y - y1) / dy
    x = x1 + t * (x2 - x1)
    return x


def compute_center_offset_px(image, avgLines, y_eval=None):
    h, w = image.shape[:2]
    if y_eval is None:
        y_eval = int(h * 0.9)  # near the bottom of the image

    if avgLines is None or len(avgLines) < 2:
        return None, None, w / 2

    # Your avgSlopeIntercept appends left then right when both exist
    leftLine = avgLines[0]
    rightLine = avgLines[1]

    xl = x_at_y(leftLine, y_eval)
    xr = x_at_y(rightLine, y_eval)

    if xl is None or xr is None:
        return None, None, w / 2

    lane_center = (xl + xr) / 2.0
    vehicle_center = w / 2.0
    offset_px = vehicle_center - lane_center
    return offset_px, lane_center, vehicle_center

##################### Main code #####################
def main():
    # Import video
    cap = cv2.VideoCapture(r"data\GRME0193.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # run the edge detection function
        edges = canny(frame)
        # Crop to the region of interest
        cropped = regionOfInterest(edges)

        # find lines with the rules set up above
        lines = cv2.HoughLinesP(
            cropped,
            HOUGH_RHO,
            HOUGH_THETA,
            HOUGH_THRESH,
            minLineLength=HOUGH_MIN_LEN,
            maxLineGap=HOUGH_MAX_GAP
        )


        avgLines = avgSlopeIntercept(frame, lines)
        lineImage = displayLines(frame, avgLines)

        combo = cv2.addWeighted(frame, 0.9, lineImage, 1.0, 1.0)

        # work out the center of the two found road lanes
        offset_px, lane_center_x, vehicle_center_x = compute_center_offset_px(frame, avgLines)

        THRESH_PX = 25

        # monitoring the position of the car
        if offset_px is None:
            status = "Lane center: NOT AVAILABLE"
        else:
            if abs(offset_px) <= THRESH_PX:
                status = f"Centered (offset {offset_px:+.0f}px)"
            elif offset_px > 0:
                status = f"Off-center: RIGHT ({offset_px:+.0f}px)"
            else:
                status = f"Off-center: LEFT ({offset_px:+.0f}px)"

        # Draw text
        cv2.putText(combo, status, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

        # Optional: draw center markers for debugging
        h, w = frame.shape[:2]
        cv2.line(combo, (int(w/2), h), (int(w/2), int(h*0.8)), (255, 255, 255), 2)
        if lane_center_x is not None:
            cv2.line(combo, (int(lane_center_x), h), (int(lane_center_x), int(h*0.8)), (0, 255, 255), 2)

        cv2.imshow("Result", combo)
        cv2.imshow("Canny", edges)
        cv2.imshow("ROI", cropped)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

##################### 
if __name__ == "__main__":
    main()

