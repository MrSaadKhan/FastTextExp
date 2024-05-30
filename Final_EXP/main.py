# main.py
import os
from result_builder import build_results
from itertools import combinations_with_replacement
from tqdm import tqdm

classifier_option = 1 # 0 for calculate_similarity, 1 for SVC classifier
bert_option = 1

if bert_option == 1:
    import BERT_new
else: 
    from FastText_function_new import calculate_similarity

if classifier_option != 0:
    import classifier

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
    
    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(device_combinations))

    # Iterate through each combination
    for dev1, dev2 in device_combinations:
        # print(dev1, dev2)
        if classifier_option == 0:
            mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen = BERT_new.bert(directory_path, dev1, dev2, classifier_option)
            build_results(device_indices, dev1, dev2, mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen)
        
        if classifier_option == 1:
            ## SVC classifier commands
            BERT_new.bert(directory_path, dev1, dev2, classifier_option)
            
        # Update progress bar
        progress_bar.update(1)
        # progress_bar.set_description(f"\033[33mProgress: {progress_bar.n / len(device_combinations) * 100:.2f}%\033[33m")
        progress_bar.set_description(f"\033[33mProgress: {progress_bar.n / len(device_combinations) * 100:.2f}%\033[m")

    # print(device_indices)
    # Close progress bar
    progress_bar.close()

if __name__ == "__main__":
    # Directory path to read files from
    directory_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(directory_path):
        directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    main(directory_path)