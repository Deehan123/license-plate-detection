import pandas as pd
import re
from collections import Counter

input_file = "output/ocr_results.csv"
output_file = "output/final_results.csv"


def clean_text(text):
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


df = pd.read_csv(input_file)

numbers = []

for value in df["Detected Number"]:

    value = clean_text(value)

    if value != "" and value != "NOTDETECTED":
        numbers.append(value)


if len(numbers) == 0:

    print("No valid license numbers found.")
    exit()


print("--------------------------------")
print("ALL OCR RESULTS")

for number in numbers:
    print(number)

print("--------------------------------")


# Find the most common length
length_count = Counter(len(x) for x in numbers)

best_length = length_count.most_common(1)[0][0]

valid_numbers = [
    x for x in numbers
    if len(x) == best_length
]


print("Most common length:", best_length)


# Character-by-character voting

final_number = ""

for position in range(best_length):

    characters = []

    for number in valid_numbers:

        characters.append(number[position])

    count = Counter(characters)

    character = count.most_common(1)[0][0]

    final_number += character


print("--------------------------------")
print("FINAL LICENSE NUMBER")
print(final_number)
print("--------------------------------")


result = pd.DataFrame({
    "License Number": [final_number],
    "Frames Used": [len(valid_numbers)]
})

result.to_csv(
    output_file,
    index=False
)

print("Saved to:", output_file)