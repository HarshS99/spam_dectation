"""
Naive Bayes Classifier Implementation from Scratch
Bhai yeh pure Python mein likha hai - koi sklearn nahi!
"""

import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple
from config import NAIVE_BAYES_CONFIG


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes Classifier for Text Classification
    
    Yeh classifier Bayes Theorem use karta hai:
    P(class|document) = P(document|class) * P(class) / P(document)
    """
    
    def __init__(self, smoothing: float = None):
        self.smoothing = smoothing or NAIVE_BAYES_CONFIG["smoothing"]
        self.min_word_length = NAIVE_BAYES_CONFIG["min_word_length"]
        
       
        self.class_priors = {}
        
        
        self.word_likelihoods = defaultdict(lambda: defaultdict(float))
        
        
        self.vocabulary = set()
        
        
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_total_words = defaultdict(int)
        
      
        self.classes = []
        
        
        self.is_trained = False
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Text ko clean aur tokenize karta hai
        
        Steps:
        1. Lowercase conversion
        2. Special characters remove
        3. Tokenization
        4. Short words filter
        """
        
        text = text.lower()
        
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
      
        words = text.split()
        
       
        words = [w for w in words if len(w) >= self.min_word_length]
        
        return words
    
    def fit(self, emails: List[str], labels: List[str]) -> Dict:
        """
        Model ko train karta hai
        
        Args:
            emails: List of email texts
            labels: List of labels (spam/not_spam)
            
        Returns:
            Training statistics
        """
        print("🎯 Training Naive Bayes Classifier...")
        
       
        self.classes = list(set(labels))
        
       
        class_doc_counts = defaultdict(int)
        
        for email, label in zip(emails, labels):
            words = self.preprocess_text(email)
            class_doc_counts[label] += 1
            
            for word in words:
                self.vocabulary.add(word)
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1
        
         
        total_docs = len(emails)
        for cls in self.classes:
            self.class_priors[cls] = class_doc_counts[cls] / total_docs
        
        
        vocab_size = len(self.vocabulary)
        
        for cls in self.classes:
            for word in self.vocabulary:
                
                word_count = self.class_word_counts[cls][word]
                total_words = self.class_total_words[cls]
                
                
                self.word_likelihoods[cls][word] = (
                    (word_count + self.smoothing) / 
                    (total_words + self.smoothing * vocab_size)
                )
        
        self.is_trained = True
        
    
        stats = {
            "total_documents": total_docs,
            "vocabulary_size": vocab_size,
            "classes": self.classes,
            "class_distribution": dict(class_doc_counts),
            "class_priors": self.class_priors
        }
        
        print(f"✅ Training Complete!")
        print(f"   📚 Vocabulary Size: {vocab_size}")
        print(f"   📧 Total Documents: {total_docs}")
        
        return stats
    
    def predict_single(self, email: str) -> Tuple[str, Dict[str, float]]:
        """
        Single email ko predict karta hai
        
        Returns:
            Tuple of (predicted_class, probability_scores)
        """
        if not self.is_trained:
            raise Exception("Model is not trained! Call fit() first.")
        
        words = self.preprocess_text(email)
        
        
        class_scores = {}
        
        for cls in self.classes:
           
            score = math.log(self.class_priors[cls])
            
           
            for word in words:
                if word in self.vocabulary:
                    score += math.log(self.word_likelihoods[cls][word])
                else:
                 
                    vocab_size = len(self.vocabulary)
                    unknown_prob = self.smoothing / (
                        self.class_total_words[cls] + self.smoothing * vocab_size
                    )
                    score += math.log(unknown_prob)
            
            class_scores[cls] = score
        
    
        max_score = max(class_scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in class_scores.items()}
        total = sum(exp_scores.values())
        probabilities = {cls: exp_score / total for cls, exp_score in exp_scores.items()}
        
        
        predicted_class = max(class_scores, key=class_scores.get)
        
        return predicted_class, probabilities
    
    def predict(self, emails: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Multiple emails ko predict karta hai
        """
        return [self.predict_single(email) for email in emails]
    
    def get_top_spam_words(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Top spam indicator words return karta hai
        """
        if not self.is_trained:
            return []
        
        spam_ratios = []
        for word in self.vocabulary:
            spam_prob = self.word_likelihoods.get("spam", {}).get(word, 0)
            not_spam_prob = self.word_likelihoods.get("not_spam", {}).get(word, 0)
            
            if not_spam_prob > 0:
                ratio = spam_prob / not_spam_prob
                spam_ratios.append((word, ratio))
        
        spam_ratios.sort(key=lambda x: x[1], reverse=True)
        return spam_ratios[:n]
    
    def get_model_summary(self) -> Dict:
        """
        Model ka summary return karta hai
        """
        return {
            "is_trained": self.is_trained,
            "vocabulary_size": len(self.vocabulary),
            "classes": self.classes,
            "class_priors": self.class_priors,
            "smoothing": self.smoothing,
            "top_spam_words": self.get_top_spam_words(10)
        }