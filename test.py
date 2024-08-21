import torch
from torch.nn.functional import softmax
from load_model import load_model  # Import the load_model function
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize default model (could be anything, or even load dynamically)
default_model_name = "cahya/bert-base-indonesian-522M"
tokenizer, model = load_model(default_model_name)

# Prediction function
def predict_hoax(title, content):
    if tokenizer is None or model is None:
        raise ValueError("Model and tokenizer must be loaded before prediction.")
    
    print(f"Using model: {model}")
    print(f"Using tokenizer: {tokenizer}")

    text = f"{title} [SEP] {content}"
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = 'HOAX' if pred == 1 else 'NON-HOAX'
    return label

# LIME prediction function
def predict_proba_for_lime(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).detach().cpu().numpy()
        results.append(probs[0])
    return np.array(results)

def evaluate_model_performance(corrected_df, tokenizer, model):
    # Prepare the inputs and labels
    texts = corrected_df.apply(lambda row: f"{row['Title']} [SEP] {row['Content']}", axis=1).tolist()
    true_labels = corrected_df['Final_Result'].apply(lambda x: 1 if x == 'HOAX' else 0).tolist()
    
    # Predict
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()
    
    # Debug: print some predictions and labels
    print("Predictions:", preds)
    print("True Labels:", true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, zero_division=1)
    recall = recall_score(true_labels, preds, zero_division=1)
    f1 = f1_score(true_labels, preds, zero_division=1)
    
    # Debug: print metric values
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    return accuracy, precision, recall, f1