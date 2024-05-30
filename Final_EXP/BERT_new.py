import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import random
import prepare_data
import os
from sklearn.metrics.pairwise import cosine_similarity

# import time as rolex

def create_embeddings(model, tokenizer, sentences):
    # Load pre-trained BERT tokenizer and model
    print("Creating Embeddings:")
    model.eval()

    embeddings = []

    for sentence in sentences:
        # Convert list of words to a single string sentence
        sentence_str = ' '.join(sentence)
        
        # Tokenize the sentence
        inputs = tokenizer(sentence_str, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Generate the embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the embeddings for the [CLS] token
        cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
        
        embeddings.append(cls_embedding)
    
    # Convert list of embeddings to a numpy array
    embeddings_np = np.array(embeddings)
    print('\033[92mEmbedding Created ✔\033[0m')
    return embeddings_np

def bert(file_path, dev1, dev2, classifier_option, iterations=10000):
    # classifier_option: 0 for calculate_similarity, 1 for SVC classifier
    vector_size = 768 # 768 is default

    group_option = 0
    time_group = 2
    
    num2word_option = 0

    # Load data
    dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2), group_option, time_group, num2word_option)

    # Load pre-trained BERT tokenizer and model
    print('Loading Pretrained BERT model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', hidden_size=vector_size)
    print('\033[92mModel Loaded ✔\033[0m')

    if classifier_option == 0:
        mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen = calculate_similarity(model, tokenizer, dev1, dev2, dev1_seen, dev1_unseen, dev2_seen, dev2_unseen, iterations=10000)
        return mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen
    
    else:
        import classifier
        # create embeddings of the data

        dev1_seen_embedding = create_embeddings(model, tokenizer, dev1_seen)
        dev1_unseen_embedding = create_embeddings(model, tokenizer, dev1_unseen)
        dev2_seen_embedding = create_embeddings(model, tokenizer, dev2_seen)
        dev2_unseen_embedding = create_embeddings(model, tokenizer, dev2_unseen)

        classifier.classifier(file_path, dev1, dev2, dev1_seen_embedding, dev1_unseen_embedding, dev2_seen_embedding, dev2_unseen_embedding, classifier_option)
        
def calculate_similarity(model, tokenizer, dev1, dev2, dev1_seen, dev1_unseen, dev2_seen, dev2_unseen, iterations=10000):

    # Check if sequences are empty
    if not dev1_seen or not dev2_seen:
        print(f"One or both of the sequences for {dev1} and {dev2} are empty.")
        print('\033[93mSeen Data:\033[0m')

        print("Average similarity for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        print("Standard deviation for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        
        print('\033[93mUnseen Data:\033[0m')

        print(f"Average similarity between {dev1} and {dev2}:" + '\033[91mN/A\033[0m')
        print(f"Standard deviation between {dev1} and {dev2}:" + '\033[91mN/A\033[0m')
            
        return 0, 0, 0, 0

    # Function to compute BERT embeddings for a single sentence
    def compute_embedding(sentence):
        input_ids = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            output = model(input_ids)
            embedding = output.last_hidden_state[:, 0, :].numpy()  # Extract embedding for [CLS] token
        return embedding

    # Lists to store similarity scores
    different_device_similarities_seen = []
    different_device_similarities_unseen = []

    # Perform iterations for seen data
    for _ in range(iterations):  
        # Randomly select a pair of sentences from seen data
        random_pair_index = random.randint(0, len(dev1_seen) - 1)
        random_pair_index2 = random.randint(0, len(dev2_seen) - 1)

        sentence1 = dev1_seen[random_pair_index]
        sentence2 = dev2_seen[random_pair_index2]

        # time1 = rolex.time()
        # Compute embeddings for the selected pair of sentences
        embedding1 = compute_embedding(sentence1)
        embedding2 = compute_embedding(sentence2)
        # time2 = rolex.time()
        # Calculate cosine similarity between embeddings
        similarity_score_diff_device = cosine_similarity(embedding1, embedding2)[0, 0]
        # time3 = rolex.time()
        print(similarity_score_diff_device)
        different_device_similarities_seen.append(similarity_score_diff_device)
        # print(f"Compute vectors: {time2-time1} \n Compute similarity {time3-time2}")

    # Perform iterations for unseen data
    for _ in range(iterations):  
        # Randomly select a pair of sentences from unseen data
        random_pair_index = random.randint(0, len(dev1_unseen) - 1)
        random_pair_index2 = random.randint(0, len(dev2_unseen) - 1)

        sentence1 = dev1_unseen[random_pair_index]
        sentence2 = dev2_unseen[random_pair_index2]

        # Compute embeddings for the selected pair of sentences
        embedding1 = compute_embedding(sentence1)
        embedding2 = compute_embedding(sentence2)

        # Calculate cosine similarity between embeddings
        similarity_score_diff_device = cosine_similarity(embedding1, embedding2)[0, 0]
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
