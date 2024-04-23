import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random
import prepare_data
import os

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

# Load data
dev2 = 'philips_hue_bridge.json'
dev1 = 'jvc_kenwood_cu-hb1.json'

dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = flatten_list(prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2)))

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the sentences
input_ids = tokenizer(dev1_seen+dev2_seen, padding=True, truncation=True, return_tensors="pt").input_ids

# Load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Get BERT embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Extract embeddings for [CLS] tokens

print("\033[92mOriginal BERT embeddings obtained.\033[0m")

# Compute BERT embeddings for seen and unseen data
dev1_seen_input_ids = tokenizer(dev1_seen, padding=True, truncation=True, return_tensors="pt").input_ids
dev2_seen_input_ids = tokenizer(dev2_seen, padding=True, truncation=True, return_tensors="pt").input_ids
dev1_unseen_input_ids = tokenizer(dev1_unseen, padding=True, truncation=True, return_tensors="pt").input_ids
dev2_unseen_input_ids = tokenizer(dev2_unseen, padding=True, truncation=True, return_tensors="pt").input_ids

with torch.no_grad():
    dev1_seen_outputs = model(dev1_seen_input_ids)
    dev2_seen_outputs = model(dev2_seen_input_ids)
    dev1_unseen_outputs = model(dev1_unseen_input_ids)
    dev2_unseen_outputs = model(dev2_unseen_input_ids)

dev1_seen_embeddings = dev1_seen_outputs.last_hidden_state[:, 0, :].numpy()
dev2_seen_embeddings = dev2_seen_outputs.last_hidden_state[:, 0, :].numpy()
dev1_unseen_embeddings = dev1_unseen_outputs.last_hidden_state[:, 0, :].numpy()
dev2_unseen_embeddings = dev2_unseen_outputs.last_hidden_state[:, 0, :].numpy()

print("\033[92mBERT embeddings computed for seen and unseen data.\033[0m")

# Compute similarity for seen data
print("\033[94mComputing similarity for seen data...\033[0m")
same_device_similarities_seen = []
different_device_similarities_seen = []

iterations = 1000

for i in range(iterations):
    random_index1 = random.randint(0, len(dev1_seen_embeddings) - 1)
    random_index2 = random.randint(0, len(dev1_seen_embeddings) - 1)
    similarity_score_same_device = torch.nn.functional.cosine_similarity(torch.tensor(dev1_seen_embeddings[random_index1]), torch.tensor(dev1_seen_embeddings[random_index2]), dim=0)
    same_device_similarities_seen.append(similarity_score_same_device.item())

    random_index3 = random.randint(0, len(dev2_seen_embeddings) - 1)
    similarity_score_diff_device = torch.nn.functional.cosine_similarity(torch.tensor(dev1_seen_embeddings[random_index2]), torch.tensor(dev2_seen_embeddings[random_index3]), dim=0)
    different_device_similarities_seen.append(similarity_score_diff_device.item())
    
print("\033[92mSimilarity computation for seen data completed.\033[0m")

# Compute similarity for unseen data
print("\033[94mComputing similarity for unseen data...\033[0m")
same_device_similarities_unseen = []
different_device_similarities_unseen = []

for i in range(iterations):
    random_index1 = random.randint(0, len(dev1_unseen_embeddings) - 1)
    random_index2 = random.randint(0, len(dev1_unseen_embeddings) - 1)
    similarity_score_same_device = torch.nn.functional.cosine_similarity(torch.tensor(dev1_unseen_embeddings[random_index1]), torch.tensor(dev1_unseen_embeddings[random_index2]), dim=0)
    same_device_similarities_unseen.append(similarity_score_same_device.item())

    random_index3 = random.randint(0, len(dev2_unseen_embeddings) - 1)
    similarity_score_diff_device = torch.nn.functional.cosine_similarity(torch.tensor(dev1_unseen_embeddings[random_index2]), torch.tensor(dev2_unseen_embeddings[random_index3]), dim=0)
    different_device_similarities_unseen.append(similarity_score_diff_device.item())

print("\033[92mSimilarity computation for unseen data completed.\033[0m")

# Calculate mean and standard deviation for seen data
print("\033[94mCalculating mean and standard deviation for seen data...\033[0m")
mu_same_device_seen = np.mean(same_device_similarities_seen)
sigma_same_device_seen = np.std(same_device_similarities_seen)
mu_diff_device_seen = np.mean(different_device_similarities_seen)
sigma_diff_device_seen = np.std(different_device_similarities_seen)

# Print results for seen data
print('\033[93mSeen Data:\033[0m')
print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device_seen, '\033[0m')
print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device_seen, '\033[0m')

print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device_seen, '\033[0m')
print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device_seen, '\033[0m')

# Calculate mean and standard deviation for unseen data
print("\033[94mCalculating mean and standard deviation for unseen data...\033[0m")
mu_same_device_unseen = np.mean(same_device_similarities_unseen)
sigma_same_device_unseen = np.std(same_device_similarities_unseen)
mu_diff_device_unseen = np.mean(different_device_similarities_unseen)
sigma_diff_device_unseen = np.std(different_device_similarities_unseen)

# Print results for unseen data
print('\033[93mUnseen Data:\033[0m')
print("Average similarity for 2 flows within a device:" + '\033[93m', mu_same_device_unseen, '\033[0m')
print("Standard deviation for 2 flows within a device:" + '\033[93m', sigma_same_device_unseen, '\033[0m')

print("Average similarity for 2 flows with 2 different devices:" + '\033[93m', mu_diff_device_unseen, '\033[0m')
print("Standard deviation for 2 flows with 2 different devices:" + '\033[93m', sigma_diff_device_unseen, '\033[0m')
