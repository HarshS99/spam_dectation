"""
Evaluator Agent
Kaam: Model ki performance evaluate karna
"""

from typing import List, Dict

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig

from config import MODEL_CONFIG


class EvaluatorAgent:
    """
    Model Evaluation Agent
    
    Responsibilities:
    1. Calculate accuracy, precision, recall, F1
    2. Generate confusion matrix
    3. Analyze model performance
    4. Suggest improvements
    """
    
    def __init__(self):
        self.agent_name = "EvaluatorAgent"
        self.system_message = """You are a Model Evaluation Expert for spam detection.

Your responsibilities:
1. Analyze model performance metrics
2. Interpret confusion matrix results
3. Identify strengths and weaknesses
4. Suggest improvements

Be analytical and provide actionable recommendations."""

      
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
        
        self.evaluation_results = None
    
    def evaluate(
        self, 
        predictions: List[Dict], 
        true_labels: List[str]
    ) -> Dict:
        """
        Model ko evaluate karta hai
        """
        print(f"\n{'='*50}")
        print(f"📊 {self.agent_name} - Evaluating Model")
        print(f"{'='*50}")
        
      
        predicted_labels = [p["prediction"] for p in predictions]
    
        tp = tn = fp = fn = 0
        
        for pred, true in zip(predicted_labels, true_labels):
            if pred == "spam" and true == "spam":
                tp += 1
            elif pred == "not_spam" and true == "not_spam":
                tn += 1
            elif pred == "spam" and true == "not_spam":
                fp += 1
            elif pred == "not_spam" and true == "spam":
                fn += 1
        
       
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        
        self.evaluation_results = {
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "total_predictions": total,
            "correct_predictions": tp + tn,
            "incorrect_predictions": fp + fn
        }
        
        
        self._print_evaluation_results()
        
        return self.evaluation_results
    
    def _print_evaluation_results(self):
        """
        Evaluation results print karta hai
        """
        if self.evaluation_results is None:
            return
        
        cm = self.evaluation_results["confusion_matrix"]
        metrics = self.evaluation_results["metrics"]
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   ┌─────────────┬──────────────┬──────────────┐")
        print(f"   │             │ Pred: Spam   │ Pred: Not    │")
        print(f"   ├─────────────┼──────────────┼──────────────┤")
        print(f"   │ True: Spam  │ TP: {cm['true_positive']:^8} │ FN: {cm['false_negative']:^8} │")
        print(f"   │ True: Not   │ FP: {cm['false_positive']:^8} │ TN: {cm['true_negative']:^8} │")
        print(f"   └─────────────┴──────────────┴──────────────┘")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   ✅ Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   🎯 Precision: {metrics['precision']*100:.2f}%")
        print(f"   📡 Recall:    {metrics['recall']*100:.2f}%")
        print(f"   ⚖️  F1 Score:  {metrics['f1_score']*100:.2f}%")
        
        print(f"\n📋 Summary:")
        print(f"   Total: {self.evaluation_results['total_predictions']}")
        print(f"   Correct: {self.evaluation_results['correct_predictions']}")
        print(f"   Incorrect: {self.evaluation_results['incorrect_predictions']}")
    
    def get_misclassified_examples(
        self, 
        predictions: List[Dict], 
        emails: List[str], 
        true_labels: List[str]
    ) -> Dict:
        """
        Misclassified examples return karta hai
        """
        false_positives = []
        false_negatives = []
        
        for pred, email, true in zip(predictions, emails, true_labels):
            if pred["prediction"] == "spam" and true == "not_spam":
                false_positives.append({
                    "email": email,
                    "confidence": pred["confidence"]
                })
            elif pred["prediction"] == "not_spam" and true == "spam":
                false_negatives.append({
                    "email": email,
                    "confidence": pred["confidence"]
                })
        
        return {
            "false_positives": false_positives[:5],   
            "false_negatives": false_negatives[:5]   
        }
    
    def get_ai_evaluation_analysis(self) -> str:
        """
        AI se evaluation analysis leta hai
        """
        if not self.ai_enabled:
            return "AI analysis not available in offline mode."
        
        if self.evaluation_results is None:
            return "No evaluation results. Run evaluate() first."
        
        metrics = self.evaluation_results["metrics"]
        cm = self.evaluation_results["confusion_matrix"]
        
        prompt = f"""Analyze these spam detection model evaluation results:

Metrics:
- Accuracy: {metrics['accuracy']*100:.2f}%
- Precision: {metrics['precision']*100:.2f}%
- Recall: {metrics['recall']*100:.2f}%
- F1 Score: {metrics['f1_score']*100:.2f}%

Confusion Matrix:
- True Positives: {cm['true_positive']}
- True Negatives: {cm['true_negative']}
- False Positives: {cm['false_positive']}
- False Negatives: {cm['false_negative']}

Provide:
1. Overall assessment (1 sentence)
2. Key strength (1 sentence)
3. Area for improvement (1 sentence)
4. One recommendation (1 sentence)"""

        try:
            response = self.agent.step(prompt)
            return response.msgs[0].content
        except Exception as e:
            return f"AI analysis error: {str(e)}"
    
    def generate_report(self) -> str:
        """
        Complete evaluation report generate karta hai
        """
        if self.evaluation_results is None:
            return "No evaluation results available."
        
        metrics = self.evaluation_results["metrics"]
        cm = self.evaluation_results["confusion_matrix"]
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           SPAM DETECTION MODEL EVALUATION REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  CONFUSION MATRIX:                                            ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │                    Predicted                            │  ║
║  │              Spam            Not Spam                   │  ║
║  │  Actual ┌──────────────┬──────────────┐                │  ║
║  │  Spam   │ TP: {cm['true_positive']:^8} │ FN: {cm['false_negative']:^8} │                │  ║
║  │  Not    │ FP: {cm['false_positive']:^8} │ TN: {cm['true_negative']:^8} │                │  ║
║  │         └──────────────┴──────────────┘                │  ║
║  └─────────────────────────────────────────────────────────┘  ║
║                                                               ║
║  PERFORMANCE METRICS:                                         ║
║  ├─ Accuracy:   {metrics['accuracy']*100:>6.2f}%                                    ║
║  ├─ Precision:  {metrics['precision']*100:>6.2f}%                                    ║
║  ├─ Recall:     {metrics['recall']*100:>6.2f}%                                    ║
║  └─ F1 Score:   {metrics['f1_score']*100:>6.2f}%                                    ║
║                                                               ║
║  SUMMARY:                                                     ║
║  Total Predictions: {self.evaluation_results['total_predictions']:>5}                                    ║
║  Correct: {self.evaluation_results['correct_predictions']:>5}  |  Incorrect: {self.evaluation_results['incorrect_predictions']:>5}                        ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report