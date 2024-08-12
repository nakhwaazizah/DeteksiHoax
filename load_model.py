from transformers import BertTokenizer, BertForSequenceClassification

# Dictionary to map model names to their paths
model_paths = {
    "cahya/bert-base-indonesian-522M": "modelcahya",
    "indobenchmark/indobert-base-p2": "modelindobench",
    "indolem/indobert-base-uncased": "modelindolem",
    "mdhugol/indonesia-bert-sentiment-classification": "modelmdhugol"
}

# Function to load the selected model
def load_model(model_name):
    base_path = model_paths[model_name]
    weights_path = f"{base_path}/model.safetensors"
    
    # Load tokenizer and model using correct paths
    tokenizer = BertTokenizer.from_pretrained(base_path)
    model = BertForSequenceClassification.from_pretrained(base_path, weights_name=weights_path)
    model.eval()
    return tokenizer, model
