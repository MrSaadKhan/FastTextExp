from gensim.models import FastText
import os
import clean_data, prepare_data
import random
import numpy as np

def calculate_similarity(file_path, dev1, dev2, iterations=10000):
    # file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2))

    # Check if sequences are empty
    if not dev1_seen or not dev2_seen:
        print(f"One or both of the sequences for {dev1} and {dev2} are empty.")
        print('\033[93mSeen Data:\033[0m')
        print("Average similarity for 2 flows within a device:" + '\033[91mN/A\033[0m')
        print("Standard deviation for 2 flows within a device:" + '\033[91mN/A\033[0m')

        print("Average similarity for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        print("Standard deviation for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        return

    print('Creating FastText model')
    model = FastText(sentences=dev1_seen + dev2_seen, vector_size=100, window=5, min_count=1, workers=4)
    print('\033[92mFastText model created ✔\033[0m')

    # Lists to store similarity scores
    same_device_similarities = []
    different_device_similarities = []

    # Perform iterations
    for _ in range(iterations):  
        random_device1_element1 = random.choice([item[0] for item in dev1_seen])
        random_device1_element2 = random.choice([item[0] for item in dev1_seen])
        random_device2_element = random.choice([item[0] for item in dev2_seen])

        similarity_score_same_device = model.wv.similarity(random_device1_element1, random_device1_element2)
        same_device_similarities.append(similarity_score_same_device)

        similarity_score_diff_device = model.wv.similarity(random_device1_element2, random_device2_element)
        different_device_similarities.append(similarity_score_diff_device)

    mu_same_device = np.mean(same_device_similarities)
    sigma_same_device = np.std(same_device_similarities)

    mu_diff_device = np.mean(different_device_similarities)
    sigma_diff_device = np.std(different_device_similarities)

    print('\033[93mSeen Data:\033[0m')
    print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device, '\033[0m')
    print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device, '\033[0m')

    print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device, '\033[0m')
    print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device, '\033[0m')

    del same_device_similarities, different_device_similarities, random_device1_element1, random_device1_element2, random_device2_element, similarity_score_same_device, similarity_score_diff_device
    del mu_same_device, sigma_same_device, mu_diff_device, sigma_diff_device 
    same_device_similarities = []
    different_device_similarities = []

    for _ in range(iterations):
        random_device1_element1 = random.choice([item[0] for item in dev1_unseen])
        random_device1_element2 = random.choice([item[0] for item in dev1_unseen])
        random_device2_element = random.choice([item[0] for item in dev2_unseen])

        similarity_score_same_device = model.wv.similarity(random_device1_element1, random_device1_element2)
        same_device_similarities.append(similarity_score_same_device)

        similarity_score_diff_device = model.wv.similarity(random_device1_element2, random_device2_element)
        different_device_similarities.append(similarity_score_diff_device)

    mu_same_device = np.mean(same_device_similarities)
    sigma_same_device = np.std(same_device_similarities)

    mu_diff_device = np.mean(different_device_similarities)
    sigma_diff_device = np.std(different_device_similarities)

    print('\033[93mUnseen Data:\033[0m')
    print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device, '\033[0m')
    print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device, '\033[0m')

    print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device, '\033[0m')
    print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device, '\033[0m')