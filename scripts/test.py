#test script for WSL2
#since WSL2 doesnt allow direct access to webcams, I am trying to use this as a test script to stream it 
import cv2

cap = cv2.VideoCapture(0)
print("Streaming on http://localhost:8080")
print("Press Ctrl+C to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()