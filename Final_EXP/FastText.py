from gensim.models import FastText
import os
import clean_data, prepare_data
import random
import numpy as np

file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

# data = list(data_generator(file_path))
iterations = 1000
dev2 = 'philips_hue_bridge.json'
dev1 = 'jvc_kenwood_cu-hb1.json'

dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2))

# Use FastText model instead of Word2Vec
print('Creating FastText model')
model = FastText(sentences=dev1_seen + dev2_seen, vector_size=100, window=5, min_count=1, workers=4)
print('\033[92mFastText model created ✔\033[0m')

# Lists to store similarity scores
same_device_similarities = []
different_device_similarities = []

# Perform iterations
for _ in range(iterations):  # Adjust the number of iterations as needed
    # Select random elements from device 1
    random_device1_element1 = random.choice([item[0] for item in dev1_seen])
    random_device1_element2 = random.choice([item[0] for item in dev1_seen])

    # Select random element from device 2
    random_device2_element = random.choice([item[0] for item in dev2_seen])

    # Compute similarity for 2 flows within a device
    similarity_score_same_device = model.wv.similarity(random_device1_element1, random_device1_element2)
    same_device_similarities.append(similarity_score_same_device)

    # Compute similarity for 2 flows with 2 different devices
    similarity_score_diff_device = model.wv.similarity(random_device1_element2, random_device2_element)
    different_device_similarities.append(similarity_score_diff_device)

# Calculate average similarity and standard deviation for SEEN
mu_same_device = np.mean(same_device_similarities)
sigma_same_device = np.std(same_device_similarities)

mu_diff_device = np.mean(different_device_similarities)
sigma_diff_device = np.std(different_device_similarities)

# Print results
print('\033[93mSeen Data:\033[0m')
print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device, '\033[0m')
print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device, '\033[0m')

print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device, '\033[0m')
print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device, '\033[0m')


#########UNSEEN#######
del same_device_similarities, different_device_similarities, random_device1_element1, random_device1_element2, random_device2_element, similarity_score_same_device, similarity_score_diff_device
del mu_same_device, sigma_same_device, mu_diff_device, sigma_diff_device 
same_device_similarities = []
different_device_similarities = []
# Perform iterations
for _ in range(iterations):  # Adjust the number of iterations as needed
    # Select random elements from device 1
    random_device1_element1 = random.choice([item[0] for item in dev1_unseen])
    random_device1_element2 = random.choice([item[0] for item in dev1_unseen])

    # Select random element from device 2
    random_device2_element = random.choice([item[0] for item in dev2_unseen])

    # Compute similarity for 2 flows within a device
    similarity_score_same_device = model.wv.similarity(random_device1_element1, random_device1_element2)
    same_device_similarities.append(similarity_score_same_device)

    # Compute similarity for 2 flows with 2 different devices
    similarity_score_diff_device = model.wv.similarity(random_device1_element2, random_device2_element)
    different_device_similarities.append(similarity_score_diff_device)

# Calculate average similarity and standard deviation for UNSEEN
mu_same_device = np.mean(same_device_similarities)
sigma_same_device = np.std(same_device_similarities)

mu_diff_device = np.mean(different_device_similarities)
sigma_diff_device = np.std(different_device_similarities)

# Print results
print('\033[93mUnseen Data:\033[0m')
print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device, '\033[0m')
print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device, '\033[0m')

print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device, '\033[0m')
print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device, '\033[0m')