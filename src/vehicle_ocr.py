
import cv2
import pytesseract
import pandas as pd
import os
import re
from collections import Counter

# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Input files
csv_file = "output/plate_vehicle_data.csv"
plate_folder = "output/associated_plates"

# Read CSV
df = pd.read_csv(csv_file)

# Store OCR results for each vehicle
vehicle_data = {}

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

# Process each plate image
for index, row in df.iterrows():

    vehicle_id = str(row["Vehicle ID"])
    filename = row["Plate Image"]

    if vehicle_id == "UNKNOWN":
        continue

    image_path = os.path.join(plate_folder, filename)

    image = cv2.imread(image_path)

    if image is None:
        continue

    # Resize image
    image = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Remove noise
    enhanced = cv2.bilateralFilter(
        enhanced,
        9,
        75,
        75
    )

    # OCR
    text = pytesseract.image_to_string(
        enhanced,
        config=(
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
    )

    text = clean_text(text)

    print("Vehicle:", vehicle_id)
    print("Detected:", text)

    # Ignore very short readings
    if len(text) < 5:
        continue

    if vehicle_id not in vehicle_data:
        vehicle_data[vehicle_id] = []

    vehicle_data[vehicle_id].append(text)


# Select most common OCR reading for each vehicle
final_results = []

for vehicle_id, readings in vehicle_data.items():

    count = Counter(readings)

    best_number, frequency = count.most_common(1)[0]

    final_results.append({
        "Vehicle ID": vehicle_id,
        "Registration Number": best_number,
        "Frames Used": len(readings),
        "Matching Readings": frequency
    })

# Save results
result_df = pd.DataFrame(final_results)

os.makedirs("output", exist_ok=True)

result_df.to_csv(
    "output/vehicle_ocr_results.csv",
    index=False
)

print("\n==============================")
print("FINAL VEHICLE OCR RESULTS")
print("==============================")

for result in final_results:
    print("Vehicle ID:", result["Vehicle ID"])
    print("Registration Number:", result["Registration Number"])
    print("Frames Used:", result["Frames Used"])
    print("------------------------------")

print("Results saved to output/vehicle_ocr_results.csv")
