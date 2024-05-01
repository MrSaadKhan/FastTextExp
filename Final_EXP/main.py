# main.py
import os
from FastText_function_new import calculate_similarity
from result_builder import build_results
from itertools import combinations_with_replacement

def main(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Get list of files in the directory
    files = os.listdir(directory_path)

    # Create a dictionary to store device names with their indices
    device_indices = {dev: i for i, dev in enumerate(files)}

    # Generate combinations of 2 devices from file names
    device_combinations = list(combinations_with_replacement(files, 2))

    # Include the reversed combinations
    device_combinations += [(dev2, dev1) for dev1, dev2 in device_combinations if (dev1, dev2) != (dev2, dev1)]

    # Iterate through each combination
    for dev1, dev2 in device_combinations:
        # print(dev1, dev2)
        mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen = calculate_similarity(directory_path, dev1, dev2)
        build_results(device_indices, dev1, dev2, mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen)
        print()
    # print(device_indices)

if __name__ == "__main__":
    # Directory path to read files from
    # directory_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'
    directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
    main(directory_path)