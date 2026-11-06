import cv2
import pytesseract
import os
import re
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

input_folder = "output/plates"
output_folder = "output/ocr_tests"

os.makedirs(output_folder, exist_ok=True)

results = []


def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


for file in os.listdir(input_folder):

    if not file.endswith(".jpg"):
        continue

    path = os.path.join(input_folder, file)

    image = cv2.imread(path)

    if image is None:
        continue

    # Resize
    image = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Threshold 1
    thresh1 = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Threshold 2
    thresh2 = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    images = {
        "gray": gray,
        "enhanced": enhanced,
        "otsu": thresh1,
        "adaptive": thresh2
    }

    best_text = ""

    print("\n-----------------------------")
    print("Plate:", file)

    for name, img in images.items():

        output_name = file.replace(
            ".jpg",
            "_" + name + ".jpg"
        )

        cv2.imwrite(
            os.path.join(output_folder, output_name),
            img
        )

        text = pytesseract.image_to_string(
            img,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        text = clean_text(text)

        print(name, ":", text)

        if len(text) > len(best_text):
            best_text = text

    if best_text == "":
        best_text = "NOT DETECTED"

    results.append({
        "Plate Image": file,
        "Detected Number": best_text
    })


df = pd.DataFrame(results)

df.to_csv(
    "output/ocr_results.csv",
    index=False
)

print("\n=============================")
print("OCR TEST COMPLETED")
print("=============================")
print("Results saved to:")
print("output/ocr_results.csv")