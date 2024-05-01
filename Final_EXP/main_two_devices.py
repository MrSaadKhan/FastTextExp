import os
from FastText_function import calculate_similarity

def main(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Define the files to pass
    files_to_pass = ['qrio_hub.json', 'sony_bravia.json']

    # Iterate through each combination
    for dev1 in files_to_pass:
        for dev2 in files_to_pass:
            if dev1 != dev2:  # Ensure not to calculate similarity for the same file
                print(dev1, dev2)
                calculate_similarity(directory_path, dev1, dev2)
                print()

if __name__ == "__main__":
    # Directory path to read files from
    directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
    main(directory_path)
