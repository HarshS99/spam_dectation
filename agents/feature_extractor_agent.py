"""
Feature Extractor Agent
Kaam: Emails se features extract karna
"""

import re
from typing import List, Dict
from collections import Counter

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig

from config import MODEL_CONFIG


class FeatureExtractorAgent:
    """
    Feature Extraction Agent
    
    Responsibilities:
    1. Extract text features from emails
    2. Identify spam indicators
    3. Analyze word patterns
    4. Generate feature reports
    """
    
    def __init__(self):
        self.agent_name = "FeatureExtractorAgent"
        self.system_message = """You are a Feature Extraction Expert for spam detection.

Your responsibilities:
1. Identify key features that distinguish spam from legitimate emails
2. Analyze word patterns and frequencies
3. Suggest important features for classification
4. Explain why certain features are spam indicators

Be analytical and provide actionable insights."""

       
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
        
       
        self.spam_keywords = [
            'free', 'win', 'winner', 'cash', 'prize', 'money', 'urgent',
            'congratulations', 'click', 'claim', 'offer', 'limited', 'act',
            'now', 'instant', 'guarantee', 'lottery', 'million', 'dollar',
            'income', 'earn', 'profit', 'investment', 'bitcoin', 'crypto'
        ]
    
    def extract_basic_features(self, email: str) -> Dict:
        """
        Email se basic features extract karta hai
        """
        email_lower = email.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', email_lower)
        
        features = {
            "length": len(email),
            "word_count": len(words),
            "unique_words": len(set(words)),
            "has_exclamation": '!' in email,
            "exclamation_count": email.count('!'),
            "has_dollar": '$' in email,
            "has_percent": '%' in email,
            "all_caps_words": len([w for w in email.split() if w.isupper() and len(w) > 1]),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "spam_keyword_count": sum(1 for kw in self.spam_keywords if kw in email_lower),
            "spam_keywords_found": [kw for kw in self.spam_keywords if kw in email_lower]
        }
        
        return features
    
    def analyze_dataset_features(
        self, 
        spam_emails: List[str], 
        not_spam_emails: List[str]
    ) -> Dict:
        """
        Poore dataset ke features analyze karta hai
        """
        print(f"\n{'='*50}")
        print(f"🔍 {self.agent_name} - Analyzing Features")
        print(f"{'='*50}")
        
     
        spam_features = [self.extract_basic_features(e) for e in spam_emails]
        not_spam_features = [self.extract_basic_features(e) for e in not_spam_emails]
        
       
        analysis = {
            "spam": self._aggregate_features(spam_features),
            "not_spam": self._aggregate_features(not_spam_features),
            "distinctive_patterns": []
        }
        
       
        spam_avg_excl = analysis["spam"]["avg_exclamation_count"]
        not_spam_avg_excl = analysis["not_spam"]["avg_exclamation_count"]
        
        if spam_avg_excl > not_spam_avg_excl * 1.5:
            analysis["distinctive_patterns"].append(
                f"Spam emails have {spam_avg_excl:.1f}x more exclamation marks"
            )
        
        spam_keyword_avg = analysis["spam"]["avg_spam_keyword_count"]
        not_spam_keyword_avg = analysis["not_spam"]["avg_spam_keyword_count"]
        
        if spam_keyword_avg > not_spam_keyword_avg * 2:
            analysis["distinctive_patterns"].append(
                f"Spam emails contain {spam_keyword_avg:.1f}x more spam keywords"
            )
        
        
        print(f"\n📊 Feature Analysis Results:")
        print(f"   Spam emails avg length: {analysis['spam']['avg_length']:.1f}")
        print(f"   Not-spam emails avg length: {analysis['not_spam']['avg_length']:.1f}")
        print(f"   Spam keyword ratio (spam): {spam_keyword_avg:.2f}")
        print(f"   Spam keyword ratio (not-spam): {not_spam_keyword_avg:.2f}")
        
        return analysis
    
    def _aggregate_features(self, features_list: List[Dict]) -> Dict:
        """
        Features ko aggregate karta hai
        """
        if not features_list:
            return {}
        
        n = len(features_list)
        
        return {
            "avg_length": sum(f["length"] for f in features_list) / n,
            "avg_word_count": sum(f["word_count"] for f in features_list) / n,
            "avg_exclamation_count": sum(f["exclamation_count"] for f in features_list) / n,
            "avg_spam_keyword_count": sum(f["spam_keyword_count"] for f in features_list) / n,
            "has_dollar_percentage": sum(1 for f in features_list if f["has_dollar"]) / n * 100,
            "common_spam_keywords": self._get_common_keywords(features_list)
        }
    
    def _get_common_keywords(self, features_list: List[Dict]) -> List[str]:
        """
        Most common spam keywords find karta hai
        """
        all_keywords = []
        for f in features_list:
            all_keywords.extend(f.get("spam_keywords_found", []))
        
        counter = Counter(all_keywords)
        return [kw for kw, _ in counter.most_common(5)]
    
    def get_ai_feature_insights(self, analysis: Dict) -> str:
        """
        AI se feature insights leta hai
        """
        if not self.ai_enabled:
            return "AI insights not available in offline mode."
        
        prompt = f"""Based on this feature analysis of spam vs not-spam emails:

Spam emails:
- Average length: {analysis['spam']['avg_length']:.1f} chars
- Avg exclamation marks: {analysis['spam']['avg_exclamation_count']:.1f}
- Avg spam keywords: {analysis['spam']['avg_spam_keyword_count']:.1f}
- Common keywords: {analysis['spam']['common_spam_keywords']}

Not-spam emails:
- Average length: {analysis['not_spam']['avg_length']:.1f} chars
- Avg exclamation marks: {analysis['not_spam']['avg_exclamation_count']:.1f}
- Avg spam keywords: {analysis['not_spam']['avg_spam_keyword_count']:.1f}

Provide 2-3 key insights about what features best distinguish spam from legitimate emails."""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return f"AI analysis error: {str(e)}"