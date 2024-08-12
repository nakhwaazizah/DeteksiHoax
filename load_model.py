from transformers import BertTokenizer, BertForSequenceClassification

# Dictionary to map model names to their paths
model_paths = {
    "cahya/bert-base-indonesian-522M": r"DasboardBert/cahya/model.safetensors",
    "indobenchmark/indobert-base-p2": r"DasboardBert/indobench/model.safetensors",
    "indolem/indobert-base-uncased": r"DasboardBert/indolem/model.safetensors",
    "mdhugol/indonesia-bert-sentiment-classification": r"DasboardBert/mdhugol/model.safetensors"
}

# Function to load the selected model
def load_model(model_name):
    path = model_paths[model_name]
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model
