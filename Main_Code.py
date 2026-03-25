import cv2
import numpy as np
import onnxruntime as ort
import winsound
import os
import sys
import math

# --- CONFIGURATION ---
MODEL_FILE = "emotion-ferplus-8.onnx"
TARGET_EMOTIONS = ['Sad', 'Fear']

# --- CHECK FOR MODEL ---
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: '{MODEL_FILE}' is missing. Please ensure the 33MB ONNX file is in this folder.")
    sys.exit()

# --- SETUP EMOTION ENGINE ---
print("Loading Emotion Model...")
try:
    ort_session = ort.InferenceSession(MODEL_FILE)
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sys.exit()

emotion_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- HELPER: EMOTION ANALYSIS ---
def analyze_emotion(face_img):
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        input_data = resized.reshape(1, 1, 64, 64).astype(np.float32)
        
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_data})
        scores = outputs[0][0]
        
        probs = np.exp(scores) / np.sum(np.exp(scores))
        best_idx = np.argmax(probs)
        return emotion_labels[best_idx]
    except:
        return "Neutral"

# --- HELPER: COUNT FINGERS (CONVEX HULL) ---
def count_fingers(roi):
    # 1. Convert to HSV & Filter Skin Color
    img_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Generic skin color range (Adjust these if detection is bad)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 100) 

    # 2. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the biggest contour (the hand)
        contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is big enough to be a hand
        if cv2.contourArea(contour) < 1000:
            return -1, roi # Noise

        # 3. Convex Hull & Defects
        hull = cv2.convexHull(contour)
        
        # Draw the hull (Polygon around hand)
        cv2.drawContours(roi, [hull], -1, (0, 255, 255), 2)
        cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
        
        # Calculate defects (gaps between fingers)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            finger_count = 0
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Calculate triangle sides to find angle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    # Cosine rule to find angle
                    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                    
                    # If angle < 90, it's a finger gap
                    if angle <= 90:
                        finger_count += 1
                        cv2.circle(roi, far, 5, [0, 0, 255], -1) # Mark the gap
            
            return finger_count + 1, roi # +1 because 4 gaps = 5 fingers
            
        except:
            return 0, roi # Calculation error (usually means Fist/0 fingers)
            
    return -1, roi

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("\nSystem Ready! Put your hand in the GREEN BOX.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    output = frame.copy()
    
    current_emotion = "Neutral"
    gesture_status = "No Hand"
    alert_active = False

    # --- 1. FACE DETECTION ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size > 0:
            current_emotion = analyze_emotion(face_roi)
            color = (0, 0, 255) if current_emotion in TARGET_EMOTIONS else (0, 255, 0)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            cv2.putText(output, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # --- 2. HAND GESTURE (Inside ROI Box) ---
    # Define a fixed box on the right side for the hand
    cv2.rectangle(output, (400, 50), (600, 250), (255, 0, 0), 2)
    roi = frame[50:250, 400:600] # Crop the box
    
    fingers, roi_drawn = count_fingers(roi)
    
    # Overlay the drawn ROI back onto the frame
    output[50:250, 400:600] = roi_drawn
    
    # GESTURE LOGIC
    if fingers == -1:
        gesture_status = "No Hand"
    elif fingers <= 1:
        gesture_status = "FIST (TUCK)" # 0 or 1 finger visible
    else:
        gesture_status = "OPEN PALM" # 2+ fingers visible

    # --- 3. ALERT TRIGGER ---
    # Logic: Emotion is SAD/FEAR  AND  Gesture is FIST
    if current_emotion in TARGET_EMOTIONS and gesture_status == "FIST (TUCK)":
        alert_active = True

    # --- 4. DISPLAY ---
    cv2.putText(output, f"Gesture: {gesture_status}", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    if alert_active:
        cv2.rectangle(output, (0,0), (640, 60), (0, 0, 255), -1)
        cv2.putText(output, "!!! HELP NEEDED !!!", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        try: winsound.Beep(1000, 100)
        except: pass
    else:
        cv2.putText(output, f"Emotion: {current_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Final Project - Convex Hull', output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()