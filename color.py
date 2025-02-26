import cv2
import numpy as np
from typing import Tuple, List

class Position:
    def __init__(self, center: Tuple[float, float], corners: List[Tuple[float, float]]):
        self.center = center
        self.corners = corners

class ColorDetector:
    def __init__(self, lower, upper, color, alias, n=1):
        self.lower = lower
        self.upper = upper
        self.color = tuple(map(int, color))  
        self.alias = alias
        self.n = n

    def detect(self, frame, n=None) -> Tuple[np.ndarray, List[Position], str]:
        if n is None:
            n = self.n

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
        detected_objects = []

        for contour in largest_contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = int((2 * x + w) / 2), int((2 * y + h) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 0), -1)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 2)
            position = Position((center_x, center_y), [(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
            detected_objects.append(position)

        return frame, detected_objects, self.alias

def execute(frame):
    cube_detectors = [
        ColorDetector(np.array([0, 100, 100]), np.array([10, 255, 255]), np.array([0, 0, 255]), 'red', 1),
        ColorDetector(np.array([20, 100, 100]), np.array([30, 255, 255]), np.array([0, 255, 255]), 'yellow', 1),
        ColorDetector(np.array([100, 150, 0]), np.array([140, 255, 255]), np.array([255, 0, 0]), 'blue', 1),
    ]
    
    corner_detector = ColorDetector(
        np.array([50, 150, 100]),  # Adjusted lower bound (higher saturation & value)
        np.array([80, 255, 255]),  # Keeping upper bound but ensuring vibrant greens
        np.array([0, 255, 0]), 'green', 4
    )

    frame = cv2.flip(frame, 1)
    frame, detected_corners, _ = corner_detector.detect(frame)

    if len(detected_corners) == 4:
        # Sort corners to be in a consistent order
        detected_corners = sorted(detected_corners, key=lambda c: (c.center[1], c.center[0]))

        top_left, top_right = sorted(detected_corners[:2], key=lambda c: c.center[0])
        bottom_left, bottom_right = sorted(detected_corners[2:], key=lambda c: c.center[0])

        # Expected perfect image corners
        img_h, img_w = frame.shape[:2]
        expected_corners = np.float32([[0, 0], [img_w, 0], [0, img_h], [img_w, img_h]]) # pyright: ignore

        # Actual detected green marker positions (use actual corners, not center)
        detected_points = np.float32([ #pyright: ignore
            top_left.corners[0],  
            top_right.corners[1],  
            bottom_left.corners[2],  
            bottom_right.corners[3]  
        ])

        # Compute delta shift (difference from expected image corners)
        delta_shift = detected_points - expected_corners

        # Compute transformation matrix from detected corners to normalized grid
        dst_points = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]]) # pyright: ignore
        matrix = cv2.getPerspectiveTransform(detected_points, dst_points)

        detected_objects : List[Tuple[Position, str]] = []
        for detector in cube_detectors:
            frame, objs, color = detector.detect(frame)
            objs = [(obj, color) for obj in objs]
            detected_objects.extend(objs)

        for obj in detected_objects:
            cube_x, cube_y = obj[0].center

            # Subtract delta shift from cube position
            adjusted_position = np.array([cube_x, cube_y], dtype=np.float32) - delta_shift.mean(axis=0)

            # Transform adjusted position into normalized space
            transformed_point = cv2.perspectiveTransform(np.array([[adjusted_position]], dtype=np.float32), matrix)[0][0]
            
            norm_x, norm_y = int(transformed_point[0]), int(transformed_point[1])

            obj[0].center = (norm_x, norm_y)

            # Display transformed coordinates
            cv2.putText(frame, f"{norm_x},{norm_y}", (cube_x, cube_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return frame, detected_objects

    return frame, []

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = execute(frame)
        cv2.imshow("Color Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
