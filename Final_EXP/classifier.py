import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Requires embeddings of the data - not the data itself
def classifier(file_path, dev1, dev2, dev1_seen, dev1_unseen, dev2_seen, dev2_unseen, classifier_option):
    if classifier_option == 1: # SVC classifier
        svc_classifier(file_path, dev1, dev2, dev1_seen, dev1_unseen, dev2_seen, dev2_unseen)
        return

def svc_classifier(file_path, dev1, dev2, dev1_seen, dev1_unseen, dev2_seen, dev2_unseen):
    if dev1 == dev2:
        print("Same device - Can not classify")
        return
    # Combine embeddings and create labels
    X_train = np.vstack((dev1_seen, dev2_seen))
    y_train = np.array([dev1] * len(dev1_seen) + [dev2] * len(dev2_seen))

    # Train the SVM classifier
    classifier = SVC(kernel='linear', C=1.0, random_state=42)
    print("Training Classifier:")
    classifier.fit(X_train, y_train)
    print('\033[92mClassifier Trained successfully ✔\033[0m')
    # Combine unseen embeddings
    X_unseen = np.vstack((dev1_unseen, dev2_unseen))
    # Create combined labels for the unseen data
    y_unseen = np.array([dev1] * len(dev1_unseen) + [dev2] * len(dev2_unseen))
    
    # Predict on the combined unseen data
    print("Classifying Data:")
    unseen_pred = classifier.predict(X_unseen)
    print('\033[92mComplete ✔\033[0m')
    # Print classification report for the combined unseen data
    print("Classification Report for Combined Unseen Data:\n", classification_report(y_unseen, unseen_pred))

    # Return the trained classifier and predictions
    return #classifier, unseen_pred