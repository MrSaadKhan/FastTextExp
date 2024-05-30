import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random
import prepare_data
import os
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(file_path, dev1, dev2, iterations=10000):
    """
    Compute similarity scores between seen and unseen data using BERT embeddings.

    Parameters:
        file_path (str): Path to the data directory.
        dev1 (str): Name of the first device.
        dev2 (str): Name of the second device.
        iterations (int): Number of iterations for computing similarity scores.

    Returns:
        Tuple: Tuple containing mean and standard deviation of similarity scores
               for seen and unseen data.
    """
    # Load data
    dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2), 0)

    # Load pre-trained BERT tokenizer and model
    print('Loading Pretrained BERT model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print('\033[92mModel Loaded ✔\033[0m')

    # # Function to compute BERT embeddings for a list of sentences
    # def compute_embeddings(sentences):
    #     input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").input_ids
    #     with torch.no_grad():
    #         outputs = model(input_ids)
    #         embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Extract embeddings for [CLS] tokens
    #     return embeddings

    # Function to compute BERT embeddings for a list of sentences
    def compute_embeddings(sentences):
        # Tokenize sentences and obtain attention mask
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Extract embeddings for [CLS] tokens
        return embeddings


    # Flatten the list of lists into a single list for each device
    dev1_seen_flat = [sentence for sublist in dev1_seen for sentence in sublist]
    dev1_unseen_flat = [sentence for sublist in dev1_unseen for sentence in sublist]
    dev2_seen_flat = [sentence for sublist in dev2_seen for sentence in sublist]
    dev2_unseen_flat = [sentence for sublist in dev2_unseen for sentence in sublist]

    # Compute BERT embeddings for seen and unseen data
    dev1_seen_embeddings = compute_embeddings(dev1_seen_flat)
    dev2_seen_embeddings = compute_embeddings(dev2_seen_flat)
    dev1_unseen_embeddings = compute_embeddings(dev1_unseen_flat)
    dev2_unseen_embeddings = compute_embeddings(dev2_unseen_flat)

    # Lists to store similarity scores
    different_device_similarities_seen = []
    different_device_similarities_unseen = []

    # Perform iterations for seen data
    for _ in range(iterations):  
        random_index1 = random.randint(0, len(dev1_seen_embeddings) - 1)
        random_index2 = random.randint(0, len(dev2_seen_embeddings) - 1)
        similarity_score_diff_device = cosine_similarity([dev1_seen_embeddings[random_index1]], [dev2_seen_embeddings[random_index2]])[0][0]
        different_device_similarities_seen.append(similarity_score_diff_device)

    # Perform iterations for unseen data
    for _ in range(iterations):  
        random_index1 = random.randint(0, len(dev1_unseen_embeddings) - 1)
        random_index2 = random.randint(0, len(dev2_unseen_embeddings) - 1)
        similarity_score_diff_device = cosine_similarity([dev1_unseen_embeddings[random_index1]], [dev2_unseen_embeddings[random_index2]])[0][0]
        different_device_similarities_unseen.append(similarity_score_diff_device)

    # Calculate mean and standard deviation for seen data
    mu_diff_device_seen = np.mean(different_device_similarities_seen)
    sigma_diff_device_seen = np.std(different_device_similarities_seen)

    # Calculate mean and standard deviation for unseen data
    mu_diff_device_unseen = np.mean(different_device_similarities_unseen)
    sigma_diff_device_unseen = np.std(different_device_similarities_unseen)

    # Print results
    print('\033[93mSeen Data:\033[0m')

    print(f"Average similarity between {dev1} and {dev2}:" + '\033[93m', mu_diff_device_seen, '\033[0m')
    print(f"Standard deviation between {dev1} and {dev2}:" + '\033[93m', sigma_diff_device_seen, '\033[0m')
    
    print('\033[93mUnseen Data:\033[0m')

    print(f"Average similarity between {dev1} and {dev2}:" + '\033[93m', mu_diff_device_unseen, '\033[0m')
    print(f"Standard deviation between {dev1} and {dev2}:" + '\033[93m', sigma_diff_device_unseen, '\033[0m')

    return mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen

# Example usage:
# mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen = calculate_similarity(file_path, dev1, dev2)
# print(mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen)
