"""
Student Registration Script.
Run this to add new students to the face recognition database.
Usage:
    python register.py
    python register.py --list          # list registered students
    python register.py --remove "Name" # remove a student
"""

import argparse
from src.recognition.face_detector import FaceDetector
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.student_registration import StudentRegistration


def main():
    parser = argparse.ArgumentParser(description="Student Registration")
    parser.add_argument("--list", action="store_true", help="List all registered students")
    parser.add_argument("--remove", type=str, default=None, help="Remove a student by name")
    args = parser.parse_args()

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if args.list:
        students = recognizer.list_students()
        if students:
            print(f"\nRegistered Students ({len(students)}):")
            for s in students:
                print(f"  • {s}")
        else:
            print("No students registered yet.")
        return

    if args.remove:
        if recognizer.remove_student(args.remove):
            print(f"[OK] '{args.remove}' removed from database.")
        else:
            print(f"[FAIL] '{args.remove}' not found.")
        return

    reg = StudentRegistration(detector, recognizer)
    reg.run()


if __name__ == "__main__":
    main()
