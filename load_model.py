from transformers import BertTokenizer, BertForSequenceClassification

# Dictionary to map model names to their paths
model_paths = {
    "cahya/bert-base-indonesian-522M": {
        "config_path": "modelcahya",
        "weights_path": "DasboardBert/cahya/model.safetensors"
    },
    "indobenchmark/indobert-base-p2": {
        "config_path": "modelindobench",
        "weights_path": "DasboardBert/indobench/model.safetensors"
    },
    "indolem/indobert-base-uncased": {
        "config_path": "modelindolem",
        "weights_path": "DasboardBert/indolem/model.safetensors"
    },
    "mdhugol/indonesia-bert-sentiment-classification": {
        "config_path": "modelmdhugol",
        "weights_path": "DasboardBert/mdhugol/model.safetensors"
    }
}

# Function to load the selected model
def load_model(model_name):
    paths = model_paths[model_name]
    config_path = paths['config_path']
    weights_path = paths['weights_path']

    # Load tokenizer and model using correct paths
    tokenizer = BertTokenizer.from_pretrained(config_path)
    model = BertForSequenceClassification.from_pretrained(config_path, weights_name=weights_path)
    model.eval()
    return tokenizer, model
