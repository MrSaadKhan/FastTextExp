# main.py
import os
from FastText_function import calculate_similarity
from itertools import combinations

def main(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Get list of files in the directory
    files = os.listdir(directory_path)

    # Generate combinations of 2 devices from file names
    device_combinations = list(combinations(files, 2))

    # Iterate through each combination
    for dev1, dev2 in device_combinations:
        calculate_similarity(directory_path, dev1, dev2)
        print()

if __name__ == "__main__":
    # Directory path to read files from
    directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
    main(directory_path)
