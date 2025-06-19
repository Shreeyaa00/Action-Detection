import cv2
from config import VIDEO_PATH, OUTPUT_PATH, YOLO_MODEL_PATH
from yolo_detector import PersonDetector
from action_detector import ActionClassifier
from video_utils import load_video, save_video_writer

def draw_box_with_label(frame, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    cap = load_video(VIDEO_PATH)
    detector = PersonDetector(YOLO_MODEL_PATH)
    classifier = ActionClassifier()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = save_video_writer(OUTPUT_PATH, width, height, fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        persons = detector.detect(frame)

        for bbox in persons:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[y1:y2, x1:x2]
            action = classifier.classify(person_crop)
            draw_box_with_label(frame, bbox, action)

        out.write(frame)
        cv2.imshow("Surveillance Action Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
