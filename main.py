"""
Main Entry Point for Spam Detection Project
Bhai yeh main file hai - isko run karo!
"""

import os
import sys

 

from data import get_spam_data, get_not_spam_data
from agents import (
    DataPreprocessorAgent,
    FeatureExtractorAgent,
    TrainerAgent,
    PredictorAgent,
    EvaluatorAgent
)


def print_banner():
    """Print project banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🚀 SPAM DETECTION PROJECT with CAMEL AI Agents 🚀               ║
║                                                                   ║
║   Using: Naive Bayes Algorithm + Multi-Agent Architecture         ║
║   Model: Groq LLaMA 3.3 70B                                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """
    Main function - Poora spam detection pipeline yahan hai
    """
    print_banner()
    
   
    print("\n🔧 STEP 1: Initializing CAMEL AI Agents...")
    print("─" * 60)
    
    data_agent = DataPreprocessorAgent()
    feature_agent = FeatureExtractorAgent()
    trainer_agent = TrainerAgent()
    predictor_agent = PredictorAgent()
    evaluator_agent = EvaluatorAgent()
    
    print("✅ All agents initialized successfully!")
    
   
    print("\n📊 STEP 2: Loading and Preprocessing Data...")
    print("─" * 60)
    
    spam_data = get_spam_data()
    not_spam_data = get_not_spam_data()
    
   
    validation_result = data_agent.load_and_validate_data(spam_data, not_spam_data)
    
   
    print("\n🤖 AI Data Analysis:")
    ai_analysis = data_agent.get_ai_analysis(validation_result)
    print(f"   {ai_analysis}")
    
    
    train_emails, train_labels, test_emails, test_labels = data_agent.split_data(
        validation_result["valid_emails"],
        validation_result["labels"]
    )
    
    
    print("\n🔍 STEP 3: Feature Extraction & Analysis...")
    print("─" * 60)
    
    
    spam_emails_only = [item["email"] for item in spam_data]
    not_spam_emails_only = [item["email"] for item in not_spam_data]
    
    feature_analysis = feature_agent.analyze_dataset_features(
        spam_emails_only, 
        not_spam_emails_only
    )
    
     
    print("\n🤖 AI Feature Insights:")
    feature_insights = feature_agent.get_ai_feature_insights(feature_analysis)
    print(f"   {feature_insights}")
    
    
    print("\n🎓 STEP 4: Training Naive Bayes Classifier...")
    print("─" * 60)
    
    
    print("\n📚 Understanding Naive Bayes Algorithm:")
    explanation = trainer_agent.explain_naive_bayes()
    print(f"   {explanation}")
    
    
    trainer_agent.initialize_classifier(smoothing=1.0)
    training_stats = trainer_agent.train(train_emails, train_labels)
    
     
    print("\n🤖 AI Training Explanation:")
    training_explanation = trainer_agent.get_ai_training_explanation()
    print(f"   {training_explanation}")
    
    
    print("\n🔮 STEP 5: Making Predictions on Test Set...")
    print("─" * 60)
    
     
    predictor_agent.set_classifier(trainer_agent.get_classifier())
    
    
    predictions = predictor_agent.predict_batch(test_emails)
    
    
    print("\n📧 Sample Predictions:")
    for i, pred in enumerate(predictions[:5]):
        emoji = "🚨" if pred["is_spam"] else "✅"
        print(f"\n   {emoji} Email {i+1}:")
        print(f"      Text: {pred['email']}")
        print(f"      Prediction: {pred['prediction']}")
        print(f"      Confidence: {pred['confidence']:.1f}%")
    
    
    print("\n📊 STEP 6: Evaluating Model Performance...")
    print("─" * 60)
    
    
    evaluation_results = evaluator_agent.evaluate(predictions, test_labels)
    
  
    misclassified = evaluator_agent.get_misclassified_examples(
        predictions, test_emails, test_labels
    )
    
    if misclassified["false_positives"]:
        print("\n⚠️ False Positives (Legitimate emails marked as spam):")
        for fp in misclassified["false_positives"][:2]:
            print(f"   - {fp['email'][:60]}...")
    
    if misclassified["false_negatives"]:
        print("\n⚠️ False Negatives (Spam emails marked as legitimate):")
        for fn in misclassified["false_negatives"][:2]:
            print(f"   - {fn['email'][:60]}...")
    
     
    print("\n🤖 AI Evaluation Analysis:")
    eval_analysis = evaluator_agent.get_ai_evaluation_analysis()
    print(f"   {eval_analysis}")
    
   
    print("\n" + "═" * 60)
    print(evaluator_agent.generate_report())
    
  
    print("\n🎮 STEP 7: Interactive Demo")
    print("─" * 60)
    
    demo_emails = [
        "Congratulations! You won $1 million dollars!",
        "Hi, can we schedule a meeting tomorrow at 2pm?",
        "URGENT: Verify your account now to avoid suspension!",
        "Thanks for your help with the project.",
        "FREE gift card waiting for you! Click now!"
    ]
    
    print("\n📧 Demo Predictions:")
    for email in demo_emails:
        result = predictor_agent.predict_with_explanation(email)
        emoji = "🚨 SPAM" if result["is_spam"] else "✅ NOT SPAM"
        print(f"\n   Email: \"{email}\"")
        print(f"   Result: {emoji} ({result['confidence']:.1f}% confidence)")
        if result.get("influential_words"):
            words = [w["word"] for w in result["influential_words"][:3]]
            print(f"   Key words: {', '.join(words)}")
    
   
    print("\n" + "═" * 60)
    user_input = input("\n🎯 Want to try interactive mode? (yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        predictor_agent.interactive_predict()
    else:
        print("\n👋 Thank you for using Spam Detection Project!")
        print("   Built with ❤️ using CAMEL AI Agents and Naive Bayes")


if __name__ == "__main__":
    main()