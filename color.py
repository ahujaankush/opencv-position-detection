import cv2
import numpy as np

class ColorDetector:
    def __init__(self, lower, upper, color):
        self.lower = lower
        self.upper = upper
        self.color = tuple(map(int, color))  # Convert numpy array to tuple

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = None
        max_contour_area = float('-inf')
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_contour_area:
                max_contour_area = area
                max_contour = contour

        if max_contour is not None and max_contour_area > 500:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 2)
            return frame, {'x': (2.0 * x + w) / 2.0, 'y': (2.0 * y + h) / 2.0, 'color': self.color}
        
        return frame, None

def main():
    print("Starting Color Detection...")
    cap = cv2.VideoCapture(2)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define color detectors (HSV ranges and corresponding rectangle colors)
    detectors = [
        ColorDetector(np.array([0, 100, 100]), np.array([10, 255, 255]), np.array([0, 0, 255])),  # Red
        ColorDetector(np.array([20, 100, 100]), np.array([30, 255, 255]), np.array([0, 255, 255])),  # Yellow
        ColorDetector(np.array([100, 150, 0]), np.array([140, 255, 255]), np.array([255, 0, 0]))  # Blue
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        detected_objects = []
        for detector in detectors:
            frame, obj = detector.detect(frame)
            if obj:
                detected_objects.append(obj)

        # Sort objects by position (top-left to bottom-right order)
        detected_objects = sorted(detected_objects, key=lambda c: (c["y"], c["x"]))
        
        # Display the frame
        cv2.imshow("Color Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
