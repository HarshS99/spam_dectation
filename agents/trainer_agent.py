"""
Trainer Agent
Kaam: Naive Bayes model ko train karna
"""

from typing import List, Dict

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig

from config import MODEL_CONFIG
from naive_bayes import NaiveBayesClassifier


class TrainerAgent:
    """
    Model Training Agent
    
    Responsibilities:
    1. Initialize Naive Bayes classifier
    2. Train the model on data
    3. Track training progress
    4. Provide training insights
    """
    
    def __init__(self):
        self.agent_name = "TrainerAgent"
        self.system_message = """You are a Machine Learning Training Expert specializing in Naive Bayes classifiers.

Your responsibilities:
1. Explain the training process
2. Analyze training results
3. Suggest improvements
4. Explain model parameters

Be educational and clear in your explanations."""

   
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
        
     
        self.classifier = None
        self.training_stats = None
    
    def initialize_classifier(self, smoothing: float = 1.0) -> NaiveBayesClassifier:
        """
        Classifier ko initialize karta hai
        """
        print(f"\n{'='*50}")
        print(f"🎓 {self.agent_name} - Initializing Classifier")
        print(f"{'='*50}")
        
        self.classifier = NaiveBayesClassifier(smoothing=smoothing)
        print(f"✅ Naive Bayes Classifier initialized")
        print(f"   Smoothing parameter: {smoothing}")
        
        return self.classifier
    
    def train(self, emails: List[str], labels: List[str]) -> Dict:
        """
        Model ko train karta hai
        """
        print(f"\n{'='*50}")
        print(f"🎓 {self.agent_name} - Training Model")
        print(f"{'='*50}")
        
        if self.classifier is None:
            self.initialize_classifier()
        
      
        self.training_stats = self.classifier.fit(emails, labels)
        
       
        top_spam_words = self.classifier.get_top_spam_words(10)
        self.training_stats["top_spam_indicators"] = top_spam_words
        
        print(f"\n📈 Training Statistics:")
        print(f"   Vocabulary size: {self.training_stats['vocabulary_size']}")
        print(f"   Classes: {self.training_stats['classes']}")
        print(f"   Class priors: {self.training_stats['class_priors']}")
        
        print(f"\n🔥 Top Spam Indicator Words:")
        for word, ratio in top_spam_words[:5]:
            print(f"   - '{word}': {ratio:.2f}x more likely in spam")
        
        return self.training_stats
    
    def get_classifier(self) -> NaiveBayesClassifier:
        """
        Trained classifier return karta hai
        """
        if self.classifier is None or not self.classifier.is_trained:
            raise Exception("Classifier not trained! Call train() first.")
        return self.classifier
    
    def get_ai_training_explanation(self) -> str:
        """
        AI se training explanation leta hai
        """
        if not self.ai_enabled:
            return "AI explanation not available in offline mode."
        
        if self.training_stats is None:
            return "No training stats available. Train the model first."
        
        prompt = f"""Explain the Naive Bayes training results in simple terms:

Training Results:
- Vocabulary size: {self.training_stats['vocabulary_size']} unique words
- Classes: {self.training_stats['classes']}
- Class priors: {self.training_stats['class_priors']}
- Top spam indicators: {[w for w, _ in self.training_stats.get('top_spam_indicators', [])[:5]]}

Explain:
1. What these numbers mean
2. Why certain words are spam indicators
3. How the model will use this for prediction

Keep it brief (3-4 sentences)."""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return f"AI explanation error: {str(e)}"
    
    def explain_naive_bayes(self) -> str:
        """
        Naive Bayes algorithm explain karta hai
        """
        if not self.ai_enabled:
            return self._get_offline_explanation()
        
        prompt = """Explain the Naive Bayes algorithm for spam detection in simple terms.
Include:
1. Basic concept (Bayes theorem)
2. Why it's called "Naive"
3. How it classifies emails
Keep it concise (4-5 sentences)."""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return self._get_offline_explanation()
    
    def _get_offline_explanation(self) -> str:
        """
        Offline Naive Bayes explanation
        """
        return """Naive Bayes Algorithm:
        
1. Based on Bayes Theorem: P(spam|words) = P(words|spam) * P(spam) / P(words)

2. "Naive" because it assumes words are independent of each other

3. For each email:
   - Calculate probability of being spam given the words
   - Calculate probability of being not-spam given the words
   - Choose the class with higher probability

4. Uses word frequencies learned during training to make predictions."""