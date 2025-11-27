
from emotioncv.realtime_face_emotion import run_camera
from emotioncv.realtime_voice_emotion import loop_voice_emotion


def main():
    print("=== EmotionCV Hub ===")
    print("1) Real-time facial emotion (webcam)")
    print("2) Voice emotion (microphone)")
    choice = input("Select 1 or 2: ").strip()

    if choice == "1":
        run_camera()
    elif choice == "2":
        loop_voice_emotion()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
