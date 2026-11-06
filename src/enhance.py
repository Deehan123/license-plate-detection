import cv2
import os

input_folder = "output/plates"
output_folder = "output/comparison"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)

for file in files:

    image = cv2.imread(os.path.join(input_folder, file))

    if image is None:
        continue

    # Original
    original = image.copy()

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Noise removal
    denoised = cv2.medianBlur(gray, 3)

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(denoised)

    # Sharpen
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    sharpened = cv2.addWeighted(
        contrast,
        1.5,
        cv2.GaussianBlur(contrast, (0, 0), 3),
        -0.5,
        0
    )

    # Threshold
    threshold = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Save individual stages
    name = os.path.splitext(file)[0]

    cv2.imwrite(
        f"{output_folder}/{name}_original.jpg",
        original
    )

    cv2.imwrite(
        f"{output_folder}/{name}_gray.jpg",
        gray
    )

    cv2.imwrite(
        f"{output_folder}/{name}_contrast.jpg",
        contrast
    )

    cv2.imwrite(
        f"{output_folder}/{name}_sharp.jpg",
        sharpened
    )

    cv2.imwrite(
        f"{output_folder}/{name}_threshold.jpg",
        threshold
    )

print("Comparison images created.")