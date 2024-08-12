from transformers import BertTokenizer, BertForSequenceClassification

# Dictionary to map model names to their paths
model_paths = {
    "cahya/bert-base-indonesian-522M": r"C:\Users\Lenovo\Downloads\DasboardBert\cahya",
    "indobenchmark/indobert-base-p2": r"C:\Users\Lenovo\Downloads\DasboardBert\indobench",
    "indolem/indobert-base-uncased": r"C:\Users\Lenovo\Downloads\DasboardBert\indolem",
    "mdhugol/indonesia-bert-sentiment-classification": r"C:\Users\Lenovo\Downloads\DasboardBert\mdhugol"
}

# Function to load the selected model
def load_model(model_name):
    path = model_paths[model_name]
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model
