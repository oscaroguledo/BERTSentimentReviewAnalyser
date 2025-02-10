from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

class Analyser():
    def __init__(self,model):
        self.model, self.tokenizer = self.load_model(model)
    
    def load_model(self,save_directory):
        try :
            pretrainedmodel=save_directory
            tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel)
            model = AutoModelForSequenceClassification.from_pretrained(pretrainedmodel)
        except Exception:
            pretrainedmodel="nlptown/bert-base-multilingual-uncased-sentiment"  # A valid public model
            # Save the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel)
            model = AutoModelForSequenceClassification.from_pretrained(pretrainedmodel)

            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)

            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        return model, tokenizer

    def calculatesentiment(self,sentence):
        # Tokenize the input sentence
        tokens = self.tokenizer.encode(sentence, return_tensors='pt')
        
        # Get the model prediction
        with torch.no_grad():  # Disable gradient computation for inference
            result = self.model(tokens)
        
        # Extract logits and determine the sentiment (class)
        sentiment_class = torch.argmax(result.logits) + 1
        return sentiment_class

        
# analyzer = Analyser("sentimentreivewmodel.h5")
# sentence = "This is a very bad product!"
# sentiment = analyzer.calculatesentiment(sentence)
# print(f"Sentiment: {sentiment}")