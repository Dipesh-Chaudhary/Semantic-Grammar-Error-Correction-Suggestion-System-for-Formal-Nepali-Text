"""
Configuration for Nepali Semantic-Aware GEC System
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "google/mt5-small"  # or "ai4bharat/IndicBART"
    max_length: int = 128
    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v", "k", "o"])
    task_prefix: str = "correct grammar: "

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "./outputs/nepali_gec_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True  # Better for T5/mT5
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "wandb"
    seed: int = 42

@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    raw_corpus_path: str = "./data/raw/nepali_corpus.txt"
    train_output_path: str = "./data/processed/train.jsonl"
    dev_output_path: str = "./data/processed/dev.jsonl"
    test_output_path: str = "./data/processed/test.jsonl"
    
    # Generation parameters
    num_rule_based_samples: int = 10000
    num_semantic_samples: int = 2000
    num_backtranslation_samples: int = 20000
    num_tagged_corruption_samples: int = 50000
    
    # Split ratios
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Validation
    manual_validation_sample_size: int = 500
    min_sentence_length: int = 5
    max_sentence_length: int = 100

@dataclass
class ErrorTaxonomy:
    """Nepali-specific error taxonomy"""
    
    # Orthographic errors
    RASWA_DIRGHA = "ORTH:VOWEL"  # Short/long vowel confusion
    MATRA_ERROR = "ORTH:DIACRITIC"  # Diacritic errors
    CONJUNCT_ERROR = "ORTH:CONJUNCT"  # Conjunct consonant errors
    SPACING_ERROR = "ORTH:SPACING"  # Word boundary errors
    
    # Morphological errors
    CASE_MARKER = "MORPH:CASE"  # Postposition errors
    VERB_AGREEMENT = "MORPH:VERB_AGR"  # Subject-verb agreement
    TENSE_ERROR = "MORPH:TENSE"  # Tense/aspect errors
    
    # Syntactic errors
    WORD_ORDER = "SYN:ORDER"  # Word order violations
    MISSING_WORD = "SYN:MISS"  # Missing required elements
    EXTRA_WORD = "SYN:EXTRA"  # Extra words
    
    # Semantic errors
    SELECTIONAL_PREF = "SEM:SELECT"  # Selectional preference violations
    HONORIFIC_MISMATCH = "SEM:HONOR"  # Honorific level errors
    ENTITY_TYPE = "SEM:ENTITY"  # Entity type incompatibility
    SEMANTIC_ANOMALY = "SEM:ANOMALY"  # General semantic implausibility
    
    @classmethod
    def all_error_types(cls):
        """Get all error type codes"""
        return [
            cls.RASWA_DIRGHA, cls.MATRA_ERROR, cls.CONJUNCT_ERROR, 
            cls.SPACING_ERROR, cls.CASE_MARKER, cls.VERB_AGREEMENT,
            cls.TENSE_ERROR, cls.WORD_ORDER, cls.MISSING_WORD,
            cls.EXTRA_WORD, cls.SELECTIONAL_PREF, cls.HONORIFIC_MISMATCH,
            cls.ENTITY_TYPE, cls.SEMANTIC_ANOMALY
        ]

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["gleu", "bleu", "bertscore", "exact_match"])
    human_eval_sample_size: int = 100
    num_human_annotators: int = 3
    reference_based: bool = True
    compute_error_type_metrics: bool = True

class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.error_taxonomy = ErrorTaxonomy()
        self.evaluation = EvaluationConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "evaluation": self.evaluation.__dict__,
            "device": self.device
        }