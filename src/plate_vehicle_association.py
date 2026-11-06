
from ultralytics import YOLO
import cv2
import os
import csv

# Load YOLO models
vehicle_model = YOLO("yolo11n.pt")
plate_model = YOLO("models/best.pt")

# Open CCTV video
video = cv2.VideoCapture("input/cctv.mp4")

if not video.isOpened():
    print("Error: Could not open CCTV video")
    exit()

# Create output folder
os.makedirs("output/associated_plates", exist_ok=True)

# Create CSV file
csv_file = open(
    "output/plate_vehicle_data.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Frame",
    "Vehicle ID",
    "Plate Image",
    "Plate Confidence"
])

count = 0
saved = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    count += 1

    # Track vehicles
    vehicle_results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.35,
        verbose=False
    )

    vehicles = []

    for result in vehicle_results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            vehicles.append({
                "id": int(track_id),
                "box": (x1, y1, x2, y2)
            })

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Vehicle ID: " + str(int(track_id)),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Detect number plates every 5 frames
    if count % 5 == 0:

        plate_results = plate_model(
            frame,
            verbose=False
        )

        for result in plate_results:

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.40:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                plate = frame[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                # Find plate centre
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                vehicle_id = "UNKNOWN"

                # Match plate with vehicle
                for vehicle in vehicles:

                    vx1, vy1, vx2, vy2 = vehicle["box"]

                    if (vx1 <= cx <= vx2 and
                            vy1 <= cy <= vy2):

                        vehicle_id = vehicle["id"]
                        break

                # Save cropped plate
                filename = "plate_" + str(saved) + ".jpg"

                filepath = os.path.join(
                    "output/associated_plates",
                    filename
                )

                cv2.imwrite(filepath, plate)

                # Save CSV information
                writer.writerow([
                    count,
                    vehicle_id,
                    filename,
                    round(confidence, 3)
                ])

                saved += 1

                # Draw plate rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Vehicle: " + str(vehicle_id),
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    cv2.imshow(
        "Vehicle and Plate Association",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n================================")
print("PLATE ASSOCIATION COMPLETED")
print("================================")
print("Frames processed:", count)
print("Plate images saved:", saved)
print("CSV: output/plate_vehicle_data.csv")
