"""
Semantic Validation Layer for Nepali GEC
Post-processes corrections to detect semantic anomalies
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
import numpy as np

class SemanticValidator:
    """Validate semantic plausibility of corrected sentences"""
    
    def __init__(
        self, 
        model_name: str = "IRIISNEPAL/RoBERTa_Nepali_110M",
        threshold: float = 0.5,
        device: str = None
    ):
        """
        Args:
            model_name: Pre-trained model for semantic validation
            threshold: Confidence threshold for plausibility (0-1)
            device: Device to run on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load model and tokenizer
        print(f"Loading semantic validator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For now, use pre-trained model - in practice, fine-tune on plausibility data
        # Since we don't have pre-trained classifier, we'll use rule-based + model scoring
        self.model_name = model_name
        
        # Initialize rule-based checker
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize rule-based semantic checks"""
        
        # Animate entities
        self.animate_entities = {
            'मान्छे', 'मानिस', 'केटा', 'केटी', 'बच्चा', 'शिक्षक',
            'विद्यार्थी', 'डाक्टर', 'किसान', 'आमा', 'बुबा',
            'कुकुर', 'बिरालो', 'गाई', 'भैंसी', 'चरा'
        }
        
        # Inanimate entities
        self.inanimate_entities = {
            'किताब', 'कलम', 'घर', 'कुर्सी', 'टेबल', 'गाडी',
            'झ्याल', 'ढोका', 'पानी', 'खाना', 'दूध'
        }
        
        # Agentive verbs (require animate subjects)
        self.agentive_verbs = {
            'खान्छ', 'पढ्छ', 'लेख्छ', 'बोल्छ', 'सोच्छ',
            'हिँड्छ', 'दौडन्छ', 'रुन्छ', 'हाँस्छ', 'सिक्छ'
        }
        
        # Consumption verbs (require edible objects)
        self.consumption_verbs = {'खान्छ', 'खान्छन्', 'खान्छिन्'}
        self.edible_objects = {'खाना', 'भात', 'दाल', 'तरकारी', 'रोटी', 'फलफूल', 'मासु'}
        
        # Honorific patterns
        self.formal_pronouns = {'तपाईं', 'हजुर'}
        self.informal_pronouns = {'तिमी', 'तँ'}
        self.formal_endings = {'हुनुहुन्छ', 'गर्नुहुन्छ', 'जानुहुन्छ'}
        self.informal_endings = {'छस्', 'गर्छस्', 'जान्छस्'}
    
    def check_animate_verb_agreement(self, sentence: str) -> Tuple[bool, str]:
        """Check if agentive verbs have animate subjects"""
        words = sentence.split()
        
        # Simple heuristic: first word is subject
        if len(words) < 2:
            return True, "OK"
        
        subject = words[0]
        
        # Check for agentive verbs
        has_agentive = any(verb in sentence for verb in self.agentive_verbs)
        
        if has_agentive and subject in self.inanimate_entities:
            return False, "SEM:ANIM_VIOLATION - Inanimate subject with agentive verb"
        
        return True, "OK"
    
    def check_consumption_object(self, sentence: str) -> Tuple[bool, str]:
        """Check if consumption verbs have appropriate objects"""
        words = sentence.split()
        
        has_consumption = any(verb in sentence for verb in self.consumption_verbs)
        
        if has_consumption:
            # Look for object (simple heuristic: word before verb)
            for i, word in enumerate(words):
                if any(verb in word for verb in self.consumption_verbs):
                    if i > 0:
                        potential_object = words[i-1]
                        # Check if it's clearly inedible
                        if potential_object in self.inanimate_entities - self.edible_objects:
                            return False, "SEM:OBJ_TYPE - Inedible object with consumption verb"
        
        return True, "OK"
    
    def check_honorific_consistency(self, sentence: str) -> Tuple[bool, str]:
        """Check honorific level consistency"""
        words = sentence.split()
        
        has_formal_pronoun = any(p in words for p in self.formal_pronouns)
        has_informal_pronoun = any(p in words for p in self.informal_pronouns)
        
        has_formal_ending = any(ending in sentence for ending in self.formal_endings)
        has_informal_ending = any(ending in sentence for ending in self.informal_endings)
        
        # Formal pronoun should have formal ending
        if has_formal_pronoun and has_informal_ending:
            return False, "SEM:HONOR - Formal pronoun with informal verb"
        
        # Informal pronoun should have informal ending
        if has_informal_pronoun and has_formal_ending:
            return False, "SEM:HONOR - Informal pronoun with formal verb"
        
        return True, "OK"
    
    def validate_sentence(self, sentence: str) -> Dict:
        """
        Validate semantic plausibility of a sentence
        
        Returns:
            {
                'is_plausible': bool,
                'confidence': float,
                'issues': List[str],
                'severity': str  # 'none', 'minor', 'major'
            }
        """
        issues = []
        
        # Run rule-based checks
        checks = [
            self.check_animate_verb_agreement(sentence),
            self.check_consumption_object(sentence),
            self.check_honorific_consistency(sentence)
        ]
        
        for is_valid, message in checks:
            if not is_valid:
                issues.append(message)
        
        # Determine severity
        if not issues:
            severity = 'none'
            is_plausible = True
            confidence = 1.0
        elif len(issues) == 1:
            severity = 'minor'
            is_plausible = False
            confidence = 0.3
        else:
            severity = 'major'
            is_plausible = False
            confidence = 0.1
        
        return {
            'is_plausible': is_plausible,
            'confidence': confidence,
            'issues': issues,
            'severity': severity
        }
    
    def validate_batch(self, sentences: List[str]) -> List[Dict]:
        """Validate a batch of sentences"""
        return [self.validate_sentence(sent) for sent in sentences]
    
    def filter_corrections(
        self, 
        corrections: List[str], 
        min_confidence: float = 0.5
    ) -> List[Tuple[str, Dict]]:
        """
        Filter corrections based on semantic plausibility
        
        Returns:
            List of (corrected_sentence, validation_result) tuples
        """
        results = []
        
        for correction in corrections:
            validation = self.validate_sentence(correction)
            
            if validation['confidence'] >= min_confidence:
                results.append((correction, validation))
        
        return results


class SemanticValidatorTrainer:
    """Train a neural semantic validator (optional enhancement)"""
    
    def __init__(self, base_model: str = "IRIISNEPAL/RoBERTa_Nepali_110M"):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
    
    def prepare_training_data(self, semantic_error_file: str):
        """
        Prepare training data for semantic plausibility classification
        Format: sentences labeled as plausible (1) or implausible (0)
        """
        import json
        
        data = []
        
        with open(semantic_error_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                # Original sentence is plausible
                data.append({
                    'text': item['target'],
                    'label': 1  # Plausible
                })
                
                # Error sentence is implausible (if semantic error)
                if item.get('is_semantic', False):
                    data.append({
                        'text': item['source'],
                        'label': 0  # Implausible
                    })
        
        return data
    
    def train(self, train_data: List[Dict], output_dir: str = "./semantic_validator"):
        """
        Train semantic plausibility classifier
        (Simplified - full implementation would use Trainer)
        """
        from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Prepare dataset
        dataset = Dataset.from_list(train_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            save_strategy="epoch",
        )
        
        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        trainer.save_model()
        
        print(f"Semantic validator saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = SemanticValidator()
    
    # Test sentences
    test_cases = [
        ("म खाना खान्छु", "Plausible"),  # I eat food
        ("किताबले खाना खान्छ", "Implausible - inanimate subject"),  # Book eats food
        ("तपाईं जानुहुन्छ", "Plausible"),  # You (formal) go
        ("तपाईं जान्छस्", "Implausible - honorific mismatch"),  # You (formal) go (informal)
        ("गाईले दूध दिन्छ", "Plausible"),  # Cow gives milk
        ("दूधले गाई दिन्छ", "Implausible - argument reversal"),  # Milk gives cow
    ]
    
    print("=== Semantic Validation Examples ===\n")
    
    for sentence, expected in test_cases:
        result = validator.validate_sentence(sentence)
        
        print(f"Sentence:    {sentence}")
        print(f"Expected:    {expected}")
        print(f"Plausible:   {result['is_plausible']}")
        print(f"Confidence:  {result['confidence']:.2f}")
        print(f"Severity:    {result['severity']}")
        if result['issues']:
            print(f"Issues:      {', '.join(result['issues'])}")
        print()