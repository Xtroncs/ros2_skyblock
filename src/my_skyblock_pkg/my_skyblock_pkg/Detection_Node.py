#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
from ultralytics import YOLO


class Detection_Node(Node):

    def __init__(self):
        super().__init__("Detection")

        # Load YOLO model
        self.model = YOLO(
            "/home/xtroncs/Zip_folders/train-7/weights/best.pt"
        )

        # Open camera
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam/video capture."
            )

        cv2.namedWindow(
            "YOLO Live Detection",
            cv2.WINDOW_NORMAL
        )

    def detect(self):

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.get_logger().error(
                "Failed to grab frame."
            )
            return

        results = self.model(
            frame,
            conf=0.7,
            verbose=False
        )[0]

        blue_y = []
        red_y = []

        for box in results.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < 0.7:
                continue

            class_name = self.model.names[cls_id]

            color = {
                0: (255, 0, 0),
                1: (0, 0, 255),
                2: (0, 255, 255),
            }.get(
                cls_id,
                (0, 255, 0)
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Draw class name
            cv2.putText(
                frame,
                class_name,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            # Calculate center Y
            cy = (y1 + y2) / 2

            if cls_id == 0:
                blue_y.append(cy)

            elif cls_id == 1:
                red_y.append(cy)

        # Check blue/red position
        if blue_y and red_y:

            if min(blue_y) < min(red_y):

                cv2.putText(
                    frame,
                    "rotate the box",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Display frame
        cv2.imshow(
            "YOLO Live Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        return True


def main(args=None):

    rclpy.init(args=args)

    node = Detection_Node()

    # Run YOLO detection
    try:

        while rclpy.ok():

            if not node.detect():
                break

            rclpy.spin_once(
                node,
                timeout_sec=0
            )

    except KeyboardInterrupt:
        pass

    finally:

        node.cap.release()

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
