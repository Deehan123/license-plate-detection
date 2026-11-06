from ultralytics import YOLO
import cv2
import os

model = YOLO("models/best.pt")

video = cv2.VideoCapture("input/cctv.mp4")

if not video.isOpened():
    print("Error: Could not open video")
    exit()

os.makedirs("output/plates", exist_ok=True)

count = 0
saved = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    count += 1

    # Process every 5th frame
    if count % 5 != 0:
        continue

    results = model(frame)

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            filename = f"output/plates/plate_{saved}.jpg"

            cv2.imwrite(filename, plate)

            saved += 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Plate {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Frames processed:", count)
print("Plate images saved:", saved)