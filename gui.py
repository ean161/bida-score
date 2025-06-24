import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_billiard_balls(image_path):
    """
    Detect billiard balls that are on the blue cloth inside the brown border of a pool table.
    Only detect balls that are mainly red, yellow, orange, or white with improved sensitivity.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Original image with detected balls marked and ball count
    """
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a copy for drawing results
    result_img = img_rgb.copy()
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Step 1: Detect the blue table cloth
    # Define range for blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create a mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: Find contours of the blue table area
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (should be the table)
    if not contours:
        print("No table detected!")
        return result_img, 0
        
    table_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the table area
    table_mask = np.zeros_like(blue_mask)
    cv2.drawContours(table_mask, [table_contour], 0, 255, -1)
    
    # Step 3: Create a slightly eroded mask to avoid detecting balls on the border
    eroded_table_mask = cv2.erode(table_mask, kernel, iterations=2)
    
    # Step 4: Detect billiard balls using circle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Only look for circles in the eroded table area
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=eroded_table_mask)
    
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=10,
        maxRadius=30
    )
    
    # Define color ranges in HSV for the target colors
    # Red has two ranges in HSV (wraps around 0) - expanded ranges for better sensitivity
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([155, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Orange range
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([18, 255, 255])

    
    # White range (low saturation, high value) - broader range for better detection
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    
    # Draw detected circles
    ball_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Get the color of the ball by sampling pixels inside the circle
            # Create a mask for the circle
            circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(circle_mask, center, radius - 2, 255, -1)  # Reduce radius to avoid border effects
            
            # Get the average color in the circle
            # First, get all non-zero points in the mask
            ball_region = cv2.bitwise_and(hsv, hsv, mask=circle_mask)
            non_zero_points = np.nonzero(circle_mask)
            
            if len(non_zero_points[0]) > 0:
                # Instead of just using average, check what percentage of pixels match each color
                # Extract all HSV pixels in the ball region
                ball_hsv = ball_region[non_zero_points[0], non_zero_points[1]]
                
                # Create masks for each color on all pixels
                red_mask1 = cv2.inRange(ball_hsv.reshape(-1, 1, 3), lower_red1, upper_red1)
                red_mask2 = cv2.inRange(ball_hsv.reshape(-1, 1, 3), lower_red2, upper_red2)
                yellow_mask = cv2.inRange(ball_hsv.reshape(-1, 1, 3), lower_yellow, upper_yellow)
                orange_mask = cv2.inRange(ball_hsv.reshape(-1, 1, 3), lower_orange, upper_orange)
                white_mask = cv2.inRange(ball_hsv.reshape(-1, 1, 3), lower_white, upper_white)
                
                # Calculate percentage of pixels that match each color
                total_pixels = len(non_zero_points[0])
                red_percentage = (np.count_nonzero(red_mask1) + np.count_nonzero(red_mask2)) / total_pixels
                yellow_percentage = np.count_nonzero(yellow_mask) / total_pixels
                orange_percentage = np.count_nonzero(orange_mask) / total_pixels
                white_percentage = np.count_nonzero(white_mask) / total_pixels
                
                # Use a threshold to determine if a significant portion of the ball is a particular color
                color_threshold = 0.25  # If 25% or more pixels match a color, consider it that color
                
                # Check if the ball is one of our target colors based on percentage thresholds
                # Store color percentages for debugging
                color_percentages = {
                    "Red": red_percentage,
                    "Yellow": yellow_percentage,
                    "Orange": orange_percentage,
                    "White": white_percentage
                }
                
                # Get RGB average for additional white detection
                rgb_region = cv2.bitwise_and(img_rgb, img_rgb, mask=circle_mask)
                avg_r = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 0])
                avg_g = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 1])
                avg_b = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 2])
                
                # Special case for white - check if RGB values are all high and close to each other
                is_white_by_rgb = (avg_r > 180 and avg_g > 180 and avg_b > 180 and
                                 max(abs(avg_r - avg_g), abs(avg_r - avg_b), abs(avg_g - avg_b)) < 30)
                
                if red_percentage >= color_threshold:
                    color_name = "Red"
                    draw_color = (255, 0, 0)  # RGB for red
                elif yellow_percentage >= color_threshold:
                    color_name = "Vàng"
                    draw_color = (255, 255, 0)  # RGB for yellow
                elif orange_percentage >= color_threshold:
                    color_name = "Orange"
                    draw_color = (255, 165, 0)  # RGB for orange
                elif white_percentage >= color_threshold or is_white_by_rgb:
                    color_name = "White"
                    draw_color = (255, 255, 255)  # RGB for white
                else:
                    # Output debug info to console
                    print(f"Ball at {center} skipped. RGB avg: ({avg_r:.1f}, {avg_g:.1f}, {avg_b:.1f}), " 
                          f"Color %: {color_percentages}")
                    # Skip this ball if it's not one of our target colors
                    continue
                
                # Draw the ball outline with color-specific indicators
                cv2.circle(result_img, center, radius, draw_color, 2)
                
                # Add ball color info to the image
                cv2.putText(result_img, f"{color_name} {ball_count+1}", 
                           (center[0] - radius, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                
                # Optional: Output detection info to console for debugging
                print(f"Detected ball {ball_count+1}: {color_name} at {center}")
                ball_count += 1
    
    print(f"Detected {ball_count} billiard balls with red/yellow/orange/white colors inside the table area")
    return result_img, ball_count

def process_image(image_path):
    """Process and display an image with detected billiard balls"""
    result_img, ball_count = detect_billiard_balls(image_path)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(result_img)
    plt.title(f"Phát hiện {ball_count} quả bi màu đỏ/vàng/cam/trắng")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save result image for reference
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu kết quả vào: {output_path}")

# Example usage:
# process_image('path_to_your_image.jpg')

def main():
    """Main function to process multiple images"""
    # Replace these with your actual image paths
    image_paths = [
        'cam.jpg'
    ]
    
    for i, image_path in enumerate(image_paths):
        print(f"Đang xử lí ảnh {i+1}: {image_path}")
        try:
            process_image(image_path)
        except Exception as e:
            print(f"Lỗi khi xử lí ảnh {image_path}: {e}")

if __name__ == "__main__":
    main()