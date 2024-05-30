import os
import prepare_data

def main(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Get list of files in the directory
    files = sorted(os.listdir(directory_path))

    # Iterate through the files in pairs
    i = 0
    while i < len(files) - 1:
        file_path1 = os.path.join(directory_path, files[i])
        file_path2 = os.path.join(directory_path, files[i + 1])
        print(f"Processing pair: {files[i]} and {files[i + 1]}")
        prepare_data.prepare_data(file_path1, file_path2)
        i += 2

    # If there's an odd number of files, process the last file with the second last file
    if len(files) % 2 != 0:
        file_path1 = os.path.join(directory_path, files[-2])
        file_path2 = os.path.join(directory_path, files[-1])
        print(f"Processing pair: {files[-2]} and {files[-1]}")
        prepare_data.prepare_data(file_path1, file_path2)

if __name__ == "__main__":
    # Directory path to read files from
    directory_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(directory_path):
        directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    main(directory_path)
