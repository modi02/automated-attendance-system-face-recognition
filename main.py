"""
Automated Attendance System using Face Recognition
Main entry point — starts the webcam/IP-camera stream and records attendance.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_VERBOSE'] = '0'
import cv2
import numpy as np
import argparse
import sys
from src.recognition.face_detector import FaceDetector
from src.recognition.face_recognizer import FaceRecognizer
from src.attendance.attendance_logger import AttendanceLogger
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="Automated Face Recognition Attendance System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: 0 for webcam, or RTSP/IP camera URL")
    parser.add_argument("--threshold", type=float, default=Config.RECOGNITION_THRESHOLD,
                        help="Distance threshold for face recognition (default: 0.6)")
    parser.add_argument("--register", action="store_true",
                        help="Launch student registration mode")
    parser.add_argument("--report", action="store_true",
                        help="Print today's attendance report and exit")
    args = parser.parse_args()

    logger = AttendanceLogger()

    if args.report:
        logger.print_report()
        return

    detector = FaceDetector()
    recognizer = FaceRecognizer(threshold=args.threshold)

    if args.register:
        from src.recognition.student_registration import StudentRegistration
        reg = StudentRegistration(detector, recognizer)
        reg.run()
        return

    # Determine video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print(f"[INFO] Starting attendance system. Source: {source}")
    print("[INFO] Press 'q' to quit, 's' to save a snapshot.")

    fps_counter = 0
    frame_skip = 2  # Process every N-th frame for performance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Retrying...")
            continue

        fps_counter += 1
        if fps_counter % frame_skip != 0:
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Detect faces
        faces = detector.detect(frame)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            name, confidence = recognizer.identify(face_crop)
            marked = False

            if name != "Unknown":
                marked = logger.mark_attendance(name)

            # Draw bounding box
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
            label_color = (0, 200, 0) if marked else (200, 200, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # Overlay info
        cv2.putText(frame, f"Students today: {logger.count_today()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = f"data/snapshots/snap_{fps_counter}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Snapshot saved: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    logger.save()
    print("[INFO] Attendance saved. Goodbye!")


if __name__ == "__main__":
    main()
