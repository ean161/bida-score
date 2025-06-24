import cv2
import numpy as np
import time
import os

# HSV color ranges
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Orange (including yellow)
lower_orange = np.array([11, 100, 100])
upper_orange = np.array([35, 255, 255])

# White
lower_white = np.array([0, 0, 200])       # Giảm Saturation, tăng Value
upper_white = np.array([180, 60, 255])    # Mở rộng Hue, cho phép nhiều màu hơn


# Beige (kem, vàng nhạt, nâu nhạt)
lower_beige = np.array([10, 30, 150])
upper_beige = np.array([30, 100, 255])

def detect_and_save(filename="data/cam.jpg", txt_output="data/coor.txt", image_output="data/detected.jpg"):
    frame = cv2.imread(filename)
    if frame is None:
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []

    def detect_largest_object(mask, label, color_bgr):
        nonlocal frame
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_area = 0
        largest_contour = None
        
        # Find the largest object
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_contour = cnt
        
        if largest_contour is not None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append(f"{label}|{cx}|{cy}")
                
                # Draw a rectangle around the largest object
                (x, y, w, h) = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)  # Draw a rectangle around the largest object

    # Detect red
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    detect_largest_object(mask_red, "Red", (0, 0, 255))

    # Detect orange (including yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    detect_largest_object(mask_orange, "Orange", (0, 165, 255))

    # # Detect white
    # mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # detect_largest_object(mask_white, "White", (255, 255, 255))

    # Detect white and beige together
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_beige = cv2.inRange(hsv, lower_beige, upper_beige)
    mask_combined = cv2.bitwise_or(mask_white, mask_beige)
    detect_largest_object(mask_combined, "White", (255, 255, 255))


    # Save text result
    with open(txt_output, "w") as f:
        for line in results:
            f.write(line + "\n")

    # Save image with bounding boxes around largest objects
    cv2.imwrite(image_output, frame)

# Main loop
while True:
    detect_and_save()
    time.sleep(0.1)
