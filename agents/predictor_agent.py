"""
Predictor Agent
Kaam: New emails ko classify karna
"""

from typing import List, Dict, Tuple

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig

from config import MODEL_CONFIG
from naive_bayes import NaiveBayesClassifier


class PredictorAgent:
    """
    Prediction Agent
    
    Responsibilities:
    1. Classify new emails
    2. Provide confidence scores
    3. Explain predictions
    4. Handle batch predictions
    """
    
    def __init__(self, classifier: NaiveBayesClassifier = None):
        self.agent_name = "PredictorAgent"
        self.classifier = classifier
        
        self.system_message = """You are a Spam Detection Prediction Expert.

Your responsibilities:
1. Explain why an email is classified as spam or not spam
2. Identify key words/phrases that influenced the decision
3. Provide confidence analysis
4. Suggest improvements for borderline cases

Be clear and educational in your explanations."""

  
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GROQ,
                model_type=ModelType.GROQ_LLAMA_3_3_70B,
                model_config_dict=GroqConfig(
                    temperature=MODEL_CONFIG["temperature"]
                ).as_dict(),
            )
            self.agent = ChatAgent(
                system_message=self.system_message,
                model=model
            )
            self.ai_enabled = True
        except Exception as e:
            print(f"⚠️ AI Agent initialization failed: {e}")
            self.ai_enabled = False
            self.agent = None
    
    def set_classifier(self, classifier: NaiveBayesClassifier):
        """
        Classifier set karta hai
        """
        self.classifier = classifier
    
    def predict_single(self, email: str) -> Dict:
        """
        Single email ko predict karta hai
        """
        if self.classifier is None:
            raise Exception("Classifier not set! Use set_classifier() first.")
        
        prediction, probabilities = self.classifier.predict_single(email)
        
     
        confidence = max(probabilities.values()) * 100
        
        result = {
            "email": email[:100] + "..." if len(email) > 100 else email,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "is_spam": prediction == "spam"
        }
        
        return result
    
    def predict_batch(self, emails: List[str]) -> List[Dict]:
        """
        Multiple emails ko predict karta hai
        """
        print(f"\n{'='*50}")
        print(f"🔮 {self.agent_name} - Making Predictions")
        print(f"{'='*50}")
        
        results = []
        for i, email in enumerate(emails):
            result = self.predict_single(email)
            results.append(result)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(emails)} emails...")
        
     
        spam_count = sum(1 for r in results if r["is_spam"])
        print(f"\n📊 Prediction Summary:")
        print(f"   Total emails: {len(results)}")
        print(f"   Predicted spam: {spam_count}")
        print(f"   Predicted not-spam: {len(results) - spam_count}")
        
        return results
    
    def predict_with_explanation(self, email: str) -> Dict:
        """
        Prediction with detailed explanation
        """
        result = self.predict_single(email)
        
       
        words = self.classifier.preprocess_text(email)
        influential_words = []
        
        for word in words:
            if word in self.classifier.vocabulary:
                spam_prob = self.classifier.word_likelihoods.get("spam", {}).get(word, 0)
                not_spam_prob = self.classifier.word_likelihoods.get("not_spam", {}).get(word, 0)
                
                if spam_prob > 0 and not_spam_prob > 0:
                    ratio = spam_prob / not_spam_prob
                    influential_words.append({
                        "word": word,
                        "spam_likelihood_ratio": ratio,
                        "indicates": "spam" if ratio > 1 else "not_spam"
                    })
        
    
        influential_words.sort(
            key=lambda x: abs(x["spam_likelihood_ratio"] - 1), 
            reverse=True
        )
        
        result["influential_words"] = influential_words[:5]
        result["explanation"] = self._generate_explanation(result)
        
        return result
    
    def _generate_explanation(self, result: Dict) -> str:
        """
        Prediction ke liye explanation generate karta hai
        """
        pred = result["prediction"]
        conf = result["confidence"]
        words = result.get("influential_words", [])
        
        explanation = f"This email is classified as '{pred}' with {conf:.1f}% confidence.\n"
        
        if words:
            spam_words = [w["word"] for w in words if w["indicates"] == "spam"]
            not_spam_words = [w["word"] for w in words if w["indicates"] == "not_spam"]
            
            if spam_words:
                explanation += f"Spam indicators: {', '.join(spam_words)}\n"
            if not_spam_words:
                explanation += f"Legitimate indicators: {', '.join(not_spam_words)}"
        
        return explanation
    
    def get_ai_prediction_analysis(self, email: str, result: Dict) -> str:
        """
        AI se prediction analysis leta hai
        """
        if not self.ai_enabled:
            return result.get("explanation", "AI analysis not available.")
        
        prompt = f"""Analyze this spam detection prediction:

Email: "{email}"

Prediction: {result['prediction']}
Confidence: {result['confidence']:.1f}%
Probabilities: {result['probabilities']}

Explain in 2-3 sentences:
1. Why this classification makes sense
2. What key phrases/patterns led to this decision"""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return result.get("explanation", f"AI analysis error: {str(e)}")
    
    def interactive_predict(self):
        """
        Interactive prediction mode
        """
        print(f"\n{'='*50}")
        print(f"🔮 Interactive Spam Detector")
        print(f"{'='*50}")
        print("Type an email to check if it's spam.")
        print("Type 'quit' to exit.\n")
        
        while True:
            email = input("📧 Enter email text: ").strip()
            
            if email.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not email:
                print("⚠️ Please enter some text.\n")
                continue
            
            result = self.predict_with_explanation(email)
            
            print(f"\n{'─'*40}")
            if result["is_spam"]:
                print(f"🚨 SPAM DETECTED!")
            else:
                print(f"✅ NOT SPAM")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"{'─'*40}\n")