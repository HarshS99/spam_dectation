"""
CAMEL AI Agents for Spam Detection
Har agent ka alag kaam hai!
"""

from .data_preprocessor_agent import DataPreprocessorAgent
from .feature_extractor_agent import FeatureExtractorAgent
from .trainer_agent import TrainerAgent
from .predictor_agent import PredictorAgent
from .evaluator_agent import EvaluatorAgent

__all__ = [
    "DataPreprocessorAgent",
    "FeatureExtractorAgent", 
    "TrainerAgent",
    "PredictorAgent",
    "EvaluatorAgent"
]