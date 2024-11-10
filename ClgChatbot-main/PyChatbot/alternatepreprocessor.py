import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import torch
from transformers import BertTokenizer, BertModel

# ... (other imports)

class QASystem:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter=';')
        self.questions_list = self.df['Query'].tolist()
        self.answers_list = self.df['Response'].tolist()

        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def preprocess(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        return ' '.join(lemmatized_tokens)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return embeddings

    def get_response(self, text):
        processed_text = self.preprocess(text)

        # Get BERT embeddings for the input
        input_embedding = self.get_embedding(processed_text)

        # Get BERT embeddings for all questions in the dataset
        question_embeddings = np.array([self.get_embedding(q) for q in self.questions_list])

        # Calculate cosine similarity between the input and all questions
        similarities = cosine_similarity([input_embedding], question_embeddings)[0]

        # Find the index of the most similar question
        max_similar_question_index = np.argmax(similarities)
        max_similarity = similarities[max_similar_question_index]

        if max_similarity >= 0.6:
            response = self.answers_list[max_similar_question_index]
            return response
        else:
            response = "Your query seems to be incomplete. Please provide more details"
            return response
