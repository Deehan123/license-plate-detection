
import subprocess
import sys
import os

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Programs to execute
programs = [
    ("Vehicle and Plate Detection",
     "src/plate_vehicle_association.py"),

    ("OCR and Final Results",
     "src/vehicle_ocr.py")
]

print("\n====================================")
print(" LICENSE PLATE DETECTION SYSTEM")
print("====================================")

for name, file in programs:

    print("\nStarting:", name)
    print("------------------------------------")

    file_path = os.path.join(BASE_DIR, file)

    try:
        subprocess.run(
            [sys.executable, file_path],
            cwd=BASE_DIR,
            check=True
        )

        print("\nCompleted:", name)

    except subprocess.CalledProcessError:
        print("\nERROR while running:", name)
        print("Pipeline stopped.")
        sys.exit(1)

print("\n====================================")
print(" PROJECT COMPLETED SUCCESSFULLY")
print("====================================")

print("\nFinal output:")
print("output/plate_vehicle_data.csv")
print("output/vehicle_ocr_results.csv")
