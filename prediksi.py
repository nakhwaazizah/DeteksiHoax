import torch
from torch.nn.functional import softmax
from load_model import load_model  # Import the load_model function
import numpy as np

# Initialize default model (could be anything, or even load dynamically)
default_model_name = "cahya/bert-base-indonesian-522M"
tokenizer, model = load_model(default_model_name)

# Prediction function
def predict_hoax(title, content):
    text = f"{title} [SEP] {content}"
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = 'HOAX' if pred == 1 else 'NON-HOAX'
    return label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# LIME prediction function
def predict_proba_for_lime(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).detach().cpu().numpy()
        results.append(probs[0])
    return np.array(results)
