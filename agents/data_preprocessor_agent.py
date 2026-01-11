"""
Data Preprocessor Agent
Kaam: Data ko clean aur prepare karna
"""

import os
import json
import random
from typing import List, Dict, Tuple

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig

from config import MODEL_CONFIG, PROJECT_SETTINGS


class DataPreprocessorAgent:
    """
    Data Preprocessing Agent
    
    Responsibilities:
    1. Data loading and validation
    2. Data cleaning suggestions
    3. Train-test split
    4. Data statistics generation
    """
    
    def __init__(self):
        self.agent_name = "DataPreprocessorAgent"
        self.system_message = """You are a Data Preprocessing Expert Agent for email spam detection.
        
Your responsibilities:
1. Analyze email data quality
2. Suggest data cleaning strategies
3. Identify potential issues in the dataset
4. Provide insights about data distribution

Always respond in a structured JSON format when analyzing data.
Be concise and actionable in your recommendations."""

        # Initialize CAMEL AI Agent with Groq
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
            print("   Running in offline mode...")
            self.ai_enabled = False
            self.agent = None
    
    def load_and_validate_data(
        self, 
        spam_data: List[Dict], 
        not_spam_data: List[Dict]
    ) -> Dict:
        """
        Data ko load aur validate karta hai
        """
        print(f"\n{'='*50}")
        print(f"🔧 {self.agent_name} - Loading & Validating Data")
        print(f"{'='*50}")
        
        validation_result = {
            "spam_count": len(spam_data),
            "not_spam_count": len(not_spam_data),
            "total_count": len(spam_data) + len(not_spam_data),
            "is_balanced": abs(len(spam_data) - len(not_spam_data)) < 20,
            "issues": [],
            "valid_emails": [],
            "labels": []
        }
      
        for item in spam_data:
            if "email" in item and "label" in item:
                if len(item["email"].strip()) > 0:
                    validation_result["valid_emails"].append(item["email"])
                    validation_result["labels"].append(item["label"])
                else:
                    validation_result["issues"].append("Empty email found in spam")
        
        for item in not_spam_data:
            if "email" in item and "label" in item:
                if len(item["email"].strip()) > 0:
                    validation_result["valid_emails"].append(item["email"])
                    validation_result["labels"].append(item["label"])
                else:
                    validation_result["issues"].append("Empty email found in not_spam")
        
        print(f"✅ Spam Emails: {validation_result['spam_count']}")
        print(f"✅ Not Spam Emails: {validation_result['not_spam_count']}")
        print(f"✅ Total Valid: {len(validation_result['valid_emails'])}")
        print(f"✅ Data Balanced: {validation_result['is_balanced']}")
        
        return validation_result
    
    def get_ai_analysis(self, data_stats: Dict) -> str:
        """
        AI se data analysis leta hai
        """
        if not self.ai_enabled:
            return "AI analysis not available in offline mode."
        
        prompt = f"""Analyze this email dataset for spam detection:

Dataset Statistics:
- Total emails: {data_stats['total_count']}
- Spam emails: {data_stats['spam_count']}
- Not spam emails: {data_stats['not_spam_count']}
- Is balanced: {data_stats['is_balanced']}
- Issues found: {len(data_stats.get('issues', []))}

Provide a brief analysis (2-3 sentences) about:
1. Data quality assessment
2. Any concerns for training a Naive Bayes classifier"""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return f"AI analysis error: {str(e)}"
    
    def split_data(
        self, 
        emails: List[str], 
        labels: List[str],
        test_size: float = None
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Data ko train-test mein split karta hai
        """
        test_size = test_size or PROJECT_SETTINGS["test_split"]
        
        print(f"\n📊 Splitting data (Test size: {test_size*100}%)")
        
        
        combined = list(zip(emails, labels))
        random.seed(PROJECT_SETTINGS["random_seed"])
        random.shuffle(combined)
        
        
        split_idx = int(len(combined) * (1 - test_size))
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]
        
       
        train_emails, train_labels = zip(*train_data) if train_data else ([], [])
        test_emails, test_labels = zip(*test_data) if test_data else ([], [])
        
        print(f"✅ Training set: {len(train_emails)} emails")
        print(f"✅ Test set: {len(test_emails)} emails")
        
        return list(train_emails), list(train_labels), list(test_emails), list(test_labels)
    
    def get_data_statistics(self, emails: List[str], labels: List[str]) -> Dict:
        """
        Detailed data statistics generate karta hai
        """
        stats = {
            "total_emails": len(emails),
            "avg_email_length": sum(len(e) for e in emails) / len(emails) if emails else 0,
            "max_email_length": max(len(e) for e in emails) if emails else 0,
            "min_email_length": min(len(e) for e in emails) if emails else 0,
            "label_distribution": {}
        }
        
        
        for label in set(labels):
            count = labels.count(label)
            stats["label_distribution"][label] = {
                "count": count,
                "percentage": (count / len(labels) * 100) if labels else 0
            }
        
        return stats