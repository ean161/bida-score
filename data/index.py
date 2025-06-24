import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def detect_billiard_balls(frame):
    """
    Detect billiard balls that are on the blue cloth inside the brown border of a pool table.
    Only detect balls that are mainly red, yellow, orange, or white.
    
    Args:
        frame: Input frame from video capture or screen recording
        
    Returns:
        Frame with detected balls marked and ball count
    """
    # Create a copy for drawing results
    result_img = frame.copy()
    
    # Optional: Apply contrast enhancement for better detection
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    
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
        return result_img, 0
        
    table_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the table area
    table_mask = np.zeros_like(blue_mask)
    cv2.drawContours(table_mask, [table_contour], 0, 255, -1)
    
    # Step 3: Create a slightly eroded mask to avoid detecting balls on the border
    eroded_table_mask = cv2.erode(table_mask, kernel, iterations=2)
    
    # Draw table outline for debugging (optional)
    # cv2.drawContours(result_img, [table_contour], 0, (0, 0, 255), 2)
    
    # Step 4: Detect billiard balls using circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Only look for circles in the eroded table area
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=eroded_table_mask)
    
    # Use Hough Circle Transform to detect circles - adjusted for screen captures
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=45,
        param2=18,
        minRadius=15,
        maxRadius=35
    )
    
    # Define color ranges in HSV for the target colors - adjusted for screen captures
    # Red has two ranges in HSV (wraps around 0)
    lower_red1 = np.array([0, 30, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 30, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow range
    lower_yellow = np.array([18, 80, 100])
    upper_yellow = np.array([38, 255, 255])
    
    # Orange range
    lower_orange = np.array([5, 80, 100])
    upper_orange = np.array([25, 255, 255])
    
    # White range (low saturation, high value)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 60, 255])
    
    # Draw detected circles
    ball_count = 0
    detected_colors = {"Red": 0, "Yellow": 0, "Orange": 0, "White": 0}
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Get the color of the ball by sampling pixels inside the circle
            # Create a mask for the circle
            circle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
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
                # Lower threshold for screen captures which might have color distortion
                color_threshold = 0.20  # If 20% or more pixels match a color, consider it that color
                
                # Check if the ball is one of our target colors based on percentage thresholds
                # Store color percentages for debugging
                color_percentages = {
                    "Red": red_percentage,
                    "Yellow": yellow_percentage,
                    "Orange": orange_percentage,
                    "White": white_percentage
                }
                
                # Get RGB average for additional detection methods
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_region = cv2.bitwise_and(frame_rgb, frame_rgb, mask=circle_mask)
                avg_r = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 0])
                avg_g = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 1])
                avg_b = np.mean(rgb_region[non_zero_points[0], non_zero_points[1], 2])
                
                # Special case for white - relaxed criteria for screen captures
                is_white_by_rgb = (avg_r > 170 and avg_g > 170 and avg_b > 170 and
                                 max(abs(avg_r - avg_g), abs(avg_r - avg_b), abs(avg_g - avg_b)) < 40)
                
                # Special case for red - consider faded reds in screen captures
                is_red_by_rgb = (avg_r > 150 and avg_r > 1.5*avg_g and avg_r > 1.5*avg_b)
                
                if red_percentage >= color_threshold or is_red_by_rgb:
                    color_name = "Red"
                    draw_color = (0, 0, 255)  # BGR for red
                    detected_colors["Red"] += 1
                elif yellow_percentage >= color_threshold:
                    color_name = "Yellow"
                    draw_color = (0, 255, 255)  # BGR for yellow
                    detected_colors["Yellow"] += 1
                elif orange_percentage >= color_threshold:
                    color_name = "Orange"
                    draw_color = (0, 165, 255)  # BGR for orange
                    detected_colors["Orange"] += 1
                elif white_percentage >= color_threshold or is_white_by_rgb:
                    color_name = "White"
                    draw_color = (255, 255, 255)  # BGR for white
                    detected_colors["White"] += 1
                else:
                    # Skip this ball if it's not one of our target colors
                    continue
                
                # Draw the ball outline with color-specific indicators
                cv2.circle(result_img, center, radius, draw_color, 2)
                
                # Add ball color info to the image
                cv2.putText(result_img, f"{color_name}", 
                           (center[0] - radius, center[1] - radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
                
                ball_count += 1
    
    # Display total count of balls by color
    y_pos = 30
    for color, count in detected_colors.items():
        if color == "Red":
            text_color = (0, 0, 255)  # BGR for red
        elif color == "Yellow":
            text_color = (0, 255, 255)  # BGR for yellow
        elif color == "Orange":
            text_color = (0, 165, 255)  # BGR for orange
        else:  # White
            text_color = (255, 255, 255)  # BGR for white
            
        if count > 0:
            cv2.putText(result_img, f"{color}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_pos += 25
    
    # Display total count
    cv2.putText(result_img, f"Total: {ball_count}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return result_img, ball_count

def process_video_stream():
    """Process video stream from screen capture or webOrange in real-time"""
    # Initialize video capture
    cap = cv2.VideoCapture(1)  # 1 for secondary Orangeera, 0 for primary Orangeera
    
    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Không thể mở nguồn video.")
        return
    
    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Kích thước video: {frame_width}x{frame_height}")
    
    # Create a named window for display
    cv2.namedWindow('Pool Ball Detector', cv2.WINDOW_NORMAL)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # For frame skipping to improve performance
    process_every_n_frames = 2
    frame_counter = 0
    
    try:
        while True:
            # Read a frame from the video source
            ret, frame = cap.read()
            
            # If frame reading was not successful, break the loop
            if not ret:
                print("Không thể đọc khung hình từ nguồn video.")
                break
            
            # Process only every nth frame for better performance
            if frame_counter % process_every_n_frames == 0:
                # Process the frame to detect billiard balls
                result_frame, count = detect_billiard_balls(frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:  # Update FPS every 10 frames
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps = frame_count / elapsed_time
                
                # Display FPS on the frame
                cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                           (result_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the processed frame
                cv2.imshow('Pool Ball Detector', result_frame)
            else:
                # Just show the original frame without processing
                cv2.imshow('Pool Ball Detector', frame)
            
            frame_counter += 1
            
            # Check for key press - Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Dừng quay phim, nhấn 'q'.")
                break
            elif key == ord('s'):
                # Save the current frame if 's' is pressed
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pool_detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Đã lưu khung hình: {filename}")
    
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Release the video source and close all windows when done
        cap.release()
        cv2.destroyAllWindows()
        print("Đã đóng nguồn video và thoát.")
def process_screen_capture():
    """
    Process screen capture for real-time detection.
    
    Note: This requires additional libraries like mss or pyautogui.
    Install with: pip install mss pyautogui numpy opencv-python
    """
    try:
        import mss
        import numpy as np
    except ImportError:
        print("Cần cài đặt thư viện mss: pip install mss")
        return
    
    # Create a named window for display
    cv2.namedWindow('Pool Ball Detector', cv2.WINDOW_NORMAL)
    
    # Initialize screen capture
    with mss.mss() as sct:
        # Get the screen dimensions
        monitor = sct.monitors[1]  # Primary monitor
        
        # You can define a region to capture instead of full screen
        # Example: capture a 800x600 region from the center of the screen
        width, height = 800, 600
        
        # Calculate the region to capture (centered on screen)
        region = {
            'left': monitor['left'] + (monitor['width'] - width) // 2,
            'top': monitor['top'] + (monitor['height'] - height) // 2,
            'width': width,
            'height': height
        }
        
        print(f"Chụp vùng màn hình: {region}")
        print("Nhấn 'q' để thoát, 's' để lưu khung hình hiện tại.")
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # For frame skipping to improve performance
        process_every_n_frames = 2
        frame_counter = 0
        
        while True:
            # Capture the screen region
            screen = np.array(sct.grab(region))
            
            # Convert the BGRA image to BGR (remove alpha channel)
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            
            # Process only every nth frame for better performance
            if frame_counter % process_every_n_frames == 0:
                # Process the frame to detect billiard balls
                result_frame, count = detect_billiard_balls(frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:  # Update FPS every 10 frames
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps = frame_count / elapsed_time
                
                # Display FPS on the frame
                cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                           (result_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the processed frame
                cv2.imshow('Pool Ball Detector', result_frame)
            else:
                # Just show the original frame without processing
                cv2.imshow('Pool Ball Detector', frame)
            
            frame_counter += 1
            
            # Check for key press - Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Dừng chụp màn hình, nhấn 'q'.")
                break
            elif key == ord('s'):
                # Save the current frame if 's' is pressed
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pool_detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Đã lưu khung hình: {filename}")
    
    cv2.destroyAllWindows()
    print("Thoát thành công.")

def main_menu():
    """Display menu and handle user choice"""
    print("\n===== Pool Ball Color Detector =====")
    print("1. Chụp màn hình (Screen Capture)")
    print("2. Sử dụng webOrange")
    print("3. Xử lý ảnh có sẵn")
    print("4. Thoát")
    
    choice = input("\nChọn tùy chọn (1-4): ")
    
    if choice == '1':
        print("\nĐang khởi động chế độ chụp màn hình...")
        process_screen_capture()
    elif choice == '2':
        print("\nĐang khởi động webOrange...")
        process_video_stream()
    elif choice == '3':
        image_path = input("\nNhập đường dẫn đến ảnh: ")
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Không thể đọc ảnh từ: {image_path}")
                return
            
            result_img, count = detect_billiard_balls(img)
            
            # Display the result
            cv2.imshow("Detected Balls", result_img)
            print(f"Đã phát hiện {count} quả bi màu Red/Yellow/Orange/White")
            
            # Save the result
            output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
            cv2.imwrite(output_path, result_img)
            print(f"Đã lưu kết quả vào: {output_path}")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {str(e)}")
            import traceback
            traceback.print_exc()
    elif choice == '4':
        print("\nĐang thoát chương trình...")
        return
    else:
        print("\nLựa chọn không hợp lệ. Vui lòng chọn từ 1-4.")
    
    # Return to menu after operation is complete
    input("\nNhấn Enter để quay lại menu chính...")
    main_menu()

if __name__ == "__main__":
    main_menu()