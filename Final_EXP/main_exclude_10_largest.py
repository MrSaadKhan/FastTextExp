# # main.py
# import os
# from FastText_function import calculate_similarity
# from itertools import combinations

# def main(directory_path):
#     # Check if directory exists
#     if not os.path.exists(directory_path):
#         print(f"Directory '{directory_path}' does not exist.")
#         return

#     # Get list of files in the directory
#     files = os.listdir(directory_path)

#     # Sort files based on size
#     files.sort(key=lambda x: os.path.getsize(os.path.join(directory_path, x)))

#     # Exclude the largest 10 files
#     files = files[:-10]

#     # Generate combinations of 2 devices from file names
#     device_combinations = list(combinations(files, 2))

#     # Include the reversed combinations
#     device_combinations += [(dev2, dev1) for dev1, dev2 in device_combinations]

#     # Iterate through each combination
#     for dev1, dev2 in device_combinations:
#         print(dev1, dev2)
#         calculate_similarity(directory_path, dev1, dev2)
#         print()

# if __name__ == "__main__":
#     # Directory path to read files from
#     # directory_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'
#     directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
#     main(directory_path)
# main.py
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

    # Sort files based on size
    files.sort(key=lambda x: os.path.getsize(os.path.join(directory_path, x)))

    # Exclude the largest 10 files
    files = files[:-10]

    # Generate reversed combinations of 2 devices from file names
    for dev1, dev2 in combinations(files, 2):
        # print(dev2, dev1)  # Print the reversed combination
        calculate_similarity(directory_path, dev2, dev1)  # Calculate similarity with reversed devices
        print()

if __name__ == "__main__":
    # Directory path to read files from
    # directory_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'
    directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
    main(directory_path)
