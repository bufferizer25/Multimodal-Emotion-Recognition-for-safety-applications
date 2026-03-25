import cv2
import numpy as np

def nothing(x):
    pass

# Create a window with sliders
cv2.namedWindow('Calibration')
cv2.createTrackbar('H Min', 'Calibration', 0, 179, nothing)
cv2.createTrackbar('S Min', 'Calibration', 20, 255, nothing)
cv2.createTrackbar('V Min', 'Calibration', 70, 255, nothing)
cv2.createTrackbar('H Max', 'Calibration', 20, 179, nothing)
cv2.createTrackbar('S Max', 'Calibration', 255, 255, nothing)
cv2.createTrackbar('V Max', 'Calibration', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # Focus only on the ROI box (Right side)
    roi = frame[50:250, 400:600]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Get current positions of the trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Calibration')
    s_min = cv2.getTrackbarPos('S Min', 'Calibration')
    v_min = cv2.getTrackbarPos('V Min', 'Calibration')
    h_max = cv2.getTrackbarPos('H Max', 'Calibration')
    s_max = cv2.getTrackbarPos('S Max', 'Calibration')
    v_max = cv2.getTrackbarPos('V Max', 'Calibration')

    lower_skin = np.array([h_min, s_min, v_min])
    upper_skin = np.array([h_max, s_max, v_max])

    # Create the Mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Show the original ROI and the Mask
    cv2.imshow('Original', roi)
    cv2.imshow('Mask (Make Hand White, Background Black)', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n--- CALIBRATED VALUES ---")
        print(f"lower_skin = np.array([{h_min}, {s_min}, {v_min}], dtype=np.uint8)")
        print(f"upper_skin = np.array([{h_max}, {s_max}, {v_max}], dtype=np.uint8)")
        break

cap.release()
cv2.destroyAllWindows()
