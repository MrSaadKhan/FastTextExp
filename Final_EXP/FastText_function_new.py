from gensim.models import FastText
from gensim.utils import simple_preprocess
import os
import prepare_data
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def compute_sentence_similarity(model, sentence1, sentence2):
#     """
#     Compute similarity score between two sentences using word embeddings.
    
#     Parameters:
#         model (gensim.models.FastText): Trained FastText model.
#         sentence1 (str): First sentence.
#         sentence2 (str): Second sentence.
    
#     Returns:
#         float: Similarity score between the sentences.
#     """
#     # Function to compute sentence embedding
#     def sentence_embedding(sentence, model):
#         # Tokenize sentence into words
#         words = simple_preprocess(sentence)
#         # Get word embeddings
#         word_embeddings = [model.wv[word] for word in words if word in model.wv]
#         if word_embeddings:
#             # Compute sentence embedding by averaging word embeddings
#             sentence_embedding = np.mean(word_embeddings, axis=0)
#             return sentence_embedding
#         else:
#             return None
    
#     # Compute embeddings for the sentences
#     embedding1 = sentence_embedding(sentence1, model)
#     embedding2 = sentence_embedding(sentence2, model)

#     # Check if embeddings were found for both sentences
#     if embedding1 is not None and embedding2 is not None:
#         # Compute cosine similarity between embeddings
#         similarity = cosine_similarity([embedding1], [embedding2])[0][0]
#         return similarity
#     else:
#         return None

def compute_sentence_similarity(model, sentence1, sentence2):
    """
    Compute similarity score between two sentences using word embeddings.
    
    Parameters:
        model (gensim.models.FastText): Trained FastText model.
        sentence1 (str): First sentence.
        sentence2 (str): Second sentence.
    
    Returns:
        float: Similarity score between the sentences.
    """
    # Tokenize sentences into words
    words1 = simple_preprocess(sentence1)
    words2 = simple_preprocess(sentence2)

    # Compute word embeddings for both sentences
    word_embeddings1 = [model.wv[word] if word in model.wv else model.wv.get_vector(word) for word in words1]
    word_embeddings2 = [model.wv[word] if word in model.wv else model.wv.get_vector(word) for word in words2]

    # Compute sentence embeddings
    embedding1 = np.mean(word_embeddings1, axis=0)
    embedding2 = np.mean(word_embeddings2, axis=0)

    # Compute cosine similarity between embeddings
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    
    return similarity

def calculate_similarity(file_path, dev1, dev2, iterations=10000):
    # file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'
    word_embedding_option = 1
    dev1_seen, dev1_unseen, dev2_seen, dev2_unseen = prepare_data.prepare_data(os.path.join(file_path, dev1), os.path.join(file_path, dev2))

    # Check if sequences are empty
    if not dev1_seen or not dev2_seen:
        print(f"One or both of the sequences for {dev1} and {dev2} are empty.")
        print('\033[93mSeen Data:\033[0m')

        print("Average similarity for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        print("Standard deviation for 2 flows with 2 different devices:" + '\033[91mN/A\033[0m')
        
        return 0, 0, 0, 0

    # Additional tokenization for word embeddings
    if word_embedding_option == 1:
        tokenized_sentences = [[simple_preprocess(sentence[0]) for sentence in set_of_sentences] for set_of_sentences in [dev1_seen, dev1_unseen, dev2_seen, dev2_unseen]]
        dev1_seen_word = tokenized_sentences[0]
        dev1_unseen_word = tokenized_sentences[1]
        dev2_seen_word = tokenized_sentences[2]
        dev2_unseen_word = tokenized_sentences[3]


    print('Creating FastText model')
    if word_embedding_option == 1:
        model = FastText(sentences=dev1_seen_word + dev2_seen_word, vector_size=100, window=5, min_count=1, workers=4)
    else:
        model = FastText(sentences=dev1_seen + dev2_seen, vector_size=100, window=5, min_count=1, workers=4)
    print('\033[92mFastText model created ✔\033[0m')

    # Lists to store similarity scores
    different_device_similarities = []

    # Perform iterations
    for _ in range(iterations):  
        random_device1_element1 = random.choice([item[0] for item in dev1_seen])
        random_device2_element = random.choice([item[0] for item in dev2_seen])

        if word_embedding_option == 1:
            similarity_score_diff_device = compute_sentence_similarity(model, random_device1_element1, random_device2_element)

        else:
            similarity_score_diff_device = model.wv.similarity(random_device1_element1, random_device2_element)
        
        different_device_similarities.append(similarity_score_diff_device)

    mu_diff_device_seen = np.mean(different_device_similarities)
    sigma_diff_device_seen = np.std(different_device_similarities)

    print('\033[93mSeen Data:\033[0m')

    print(f"Average similarity between {dev1} and {dev2}:" + '\033[93m', mu_diff_device_seen, '\033[0m')
    print(f"Standard deviation between {dev1} and {dev2}:" + '\033[93m', sigma_diff_device_seen, '\033[0m')

    del different_device_similarities, random_device1_element1, random_device2_element, similarity_score_diff_device
    different_device_similarities = []

    for _ in range(iterations):
        random_device1_element = random.choice([item[0] for item in dev1_unseen])
        random_device2_element = random.choice([item[0] for item in dev2_unseen])

        # similarity_score_diff_device = model.wv.similarity(random_device1_element, random_device2_element)
        # different_device_similarities.append(similarity_score_diff_device)
        
        if word_embedding_option == 1:
            similarity_score_diff_device = compute_sentence_similarity(model, random_device1_element, random_device2_element)

        else:
            similarity_score_diff_device = model.wv.similarity(random_device1_element, random_device2_element)
        
        different_device_similarities.append(similarity_score_diff_device)

    mu_diff_device_unseen = np.mean(different_device_similarities)
    sigma_diff_device_unseen = np.std(different_device_similarities)

    print('\033[93mUnseen Data:\033[0m')

    print(f"Average similarity between {dev1} and {dev2}:" + '\033[93m', mu_diff_device_unseen, '\033[0m')
    print(f"Standard deviation between {dev1} and {dev2}:" + '\033[93m', sigma_diff_device_unseen, '\033[0m')

    return mu_diff_device_seen, sigma_diff_device_seen, mu_diff_device_unseen, sigma_diff_device_unseen