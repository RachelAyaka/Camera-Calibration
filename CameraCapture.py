import cv2
from datetime import datetime
import time

cap = cv2.VideoCapture(0)
time.sleep(2.0)

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        cv2.imwrite(f"calibrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", frame)

cap.release()
cv2.destroyAllWindows()