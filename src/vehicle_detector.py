from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("input/cctv.mp4")

if not video.isOpened():
    print("Error: Could not open video")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO classes:
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if cls in [2, 3, 5, 7]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{model.names[cls]} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()