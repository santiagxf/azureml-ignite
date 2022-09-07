from os import truncate
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto import AutoModelForSequenceClassification

import logging
import torch
import pandas as pd

class HuggingFaceClassifierModel:
    def __init__(self, baseline: str, tokenizer: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or baseline)
        self.model = AutoModelForSequenceClassification.from_pretrained(baseline)

        _ = self.model.eval()

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data = data['text']
    
        inputs = self.tokenizer(list(data), padding=True, truncate=True, return_tensors='pt')
        predictions = self.model(**inputs)
        probs = torch.nn.Softmax(dim=1)(predictions.logits)
        probs = probs.detach().numpy()
        
        logging.info("[INFO] Building results with hate probabilities")
        classes = probs.argmax(axis=1)
        confidence = probs.max(axis=1)

        data = data.reset_index()
        data['rating'] = classes
        data['confidence'] = confidence

        results = data[['index', 'hate', 'confidence']].groupby('index').agg({'hate': pd.Series.mode, 'confidence': 'mean' })

        return results

    def _load_pyfunc(path):
        import os
        return HuggingFaceClassifierModel(os.path.abspath(path))