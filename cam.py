import cv2
import datetime

# Sử dụng webcam từ iPhone qua FaceTime
cap = cv2.VideoCapture(1)

# Lặp để chụp ảnh
while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()

    # Lưu ảnh với tên tệp được định dạng theo thời gian
    cv2.imwrite(f"data/cam.jpg", frame)

    # Chờ 3 giây trước khi chụp ảnh tiếp theo
    cv2.waitKey(1000)