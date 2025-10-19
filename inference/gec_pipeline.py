"""
Complete Inference Pipeline for Nepali Semantic-Aware GEC
Integrates trained model with semantic validation
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import List, Dict, Union
import json

import sys
sys.path.append('..')
from models.semantic_validator import SemanticValidator


class NepaliGECPipeline:
    """End-to-end GEC pipeline with semantic validation"""
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "google/mt5-small",
        use_semantic_validation: bool = True,
        device: str = None
    ):
        """
        Args:
            model_path: Path to trained LoRA model
            base_model: Base model name (should match training)
            use_semantic_validation: Whether to apply semantic validation
            device: Device to use (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_semantic_validation = use_semantic_validation
        
        # Load model and tokenizer
        print(f"Loading GEC model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load base model
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        
        # Load semantic validator
        if use_semantic_validation:
            print("Loading semantic validator...")
            self.semantic_validator = SemanticValidator(device=self.device)
        else:
            self.semantic_validator = None
        
        print("Pipeline ready!")
    
    def correct_sentence(
        self,
        sentence: str,
        max_length: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        temperature: float = 1.0
    ) -> Union[str, List[str]]:
        """
        Correct a single sentence
        
        Args:
            sentence: Input sentence with errors
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            num_return_sequences: Number of corrections to return
            temperature: Sampling temperature
        
        Returns:
            Corrected sentence(s)
        """
        # Prepare input
        input_text = "correct grammar: " + sentence
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        # Generate corrections
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                early_stopping=True
            )
        
        # Decode
        corrections = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # Apply semantic validation if enabled
        if self.use_semantic_validation and self.semantic_validator:
            validated_corrections = []
            
            for correction in corrections:
                validation = self.semantic_validator.validate_sentence(correction)
                
                if validation['is_plausible']:
                    validated_corrections.append(correction)
            
            # If all corrections filtered, return best one with warning
            if not validated_corrections:
                print(f"Warning: All corrections filtered by semantic validator")
                validated_corrections = corrections[:1]
            
            corrections = validated_corrections
        
        # Return single or multiple
        if num_return_sequences == 1:
            return corrections[0]
        else:
            return corrections
    
    def correct_batch(
        self,
        sentences: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """Correct a batch of sentences"""
        corrections = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Prepare inputs
            input_texts = ["correct grammar: " + s for s in batch]
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=generation_kwargs.get('max_length', 128)
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=generation_kwargs.get('max_length', 128),
                    num_beams=generation_kwargs.get('num_beams', 4),
                    early_stopping=True
                )
            
            # Decode
            batch_corrections = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            corrections.extend(batch_corrections)
        
        return corrections
    
    def correct_with_details(
        self,
        sentence: str,
        **generation_kwargs
    ) -> Dict:
        """
        Correct sentence and return detailed information
        
        Returns:
            {
                'input': str,
                'correction': str,
                'semantic_validation': Dict,
                'changes_made': bool
            }
        """
        correction = self.correct_sentence(sentence, **generation_kwargs)
        
        # Get semantic validation
        if self.use_semantic_validation:
            validation = self.semantic_validator.validate_sentence(correction)
        else:
            validation = {'is_plausible': True, 'issues': [], 'severity': 'none'}
        
        return {
            'input': sentence,
            'correction': correction,
            'semantic_validation': validation,
            'changes_made': sentence.strip() != correction.strip()
        }
    
    def interactive_mode(self):
        """Interactive correction mode"""
        print("\n" + "=" * 60)
        print("Nepali Grammar Error Correction - Interactive Mode")
        print("=" * 60)
        print("Enter sentences to correct (type 'exit' to quit)\n")
        
        while True:
            try:
                sentence = input("Input: ").strip()
                
                if sentence.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if not sentence:
                    continue
                
                # Correct with details
                result = self.correct_with_details(sentence)
                
                print(f"\nCorrection: {result['correction']}")
                
                if result['changes_made']:
                    print("✓ Changes made")
                else:
                    print("✓ No errors detected")
                
                # Show semantic validation
                val = result['semantic_validation']
                if not val['is_plausible']:
                    print(f"⚠ Semantic issues detected ({val['severity']})")
                    for issue in val['issues']:
                        print(f"  - {issue}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def process_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 8
    ):
        """Process a file of sentences"""
        print(f"Processing {input_file}...")
        
        # Load sentences
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(sentences)} sentences")
        
        # Correct in batches
        corrections = self.correct_batch(sentences, batch_size=batch_size)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for original, correction in zip(sentences, corrections):
                result = {
                    'input': original,
                    'output': correction,
                    'changed': original != correction
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Results saved to {output_file}")
        
        # Statistics
        num_changed = sum(1 for o, c in zip(sentences, corrections) if o != c)
        print(f"\nStatistics:")
        print(f"  Total sentences: {len(sentences)}")
        print(f"  Corrected: {num_changed} ({num_changed/len(sentences)*100:.1f}%)")
        print(f"  Unchanged: {len(sentences) - num_changed}")


def main():
    """Demo usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nepali GEC Inference")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--base_model', type=str, default='google/mt5-small',
                       help='Base model name')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file', 'single'],
                       default='interactive', help='Inference mode')
    parser.add_argument('--input', type=str, help='Input file or sentence')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--no_semantic', action='store_true',
                       help='Disable semantic validation')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NepaliGECPipeline(
        model_path=args.model_path,
        base_model=args.base_model,
        use_semantic_validation=not args.no_semantic
    )
    
    # Run based on mode
    if args.mode == 'interactive':
        pipeline.interactive_mode()
    
    elif args.mode == 'file':
        if not args.input or not args.output:
            print("Error: --input and --output required for file mode")
            return
        pipeline.process_file(args.input, args.output)
    
    elif args.mode == 'single':
        if not args.input:
            print("Error: --input required for single mode")
            return
        
        result = pipeline.correct_with_details(args.input)
        print(f"Input:      {result['input']}")
        print(f"Correction: {result['correction']}")
        
        val = result['semantic_validation']
        print(f"Plausible:  {val['is_plausible']}")
        if val['issues']:
            print(f"Issues:     {', '.join(val['issues'])}")


if __name__ == "__main__":
    # Example usage without args
    print("Loading example pipeline...")
    
    # Simulated model path - replace with actual trained model
    model_path = "./outputs/nepali_gec_model"
    
    try:
        pipeline = NepaliGECPipeline(
            model_path=model_path,
            use_semantic_validation=True
        )
        
        # Test corrections
        test_sentences = [
            "म किताब पढछु",  # Missing ्
            "उनी घर मा छ",  # Should be छन्
            "किताबले खाना खान्छ",  # Semantic error
        ]
        
        print("\n=== Test Corrections ===\n")
        for sent in test_sentences:
            result = pipeline.correct_with_details(sent)
            print(f"Input:  {result['input']}")
            print(f"Output: {result['correction']}")
            print(f"Valid:  {result['semantic_validation']['is_plausible']}")
            print()
        
    except Exception as e:
        print(f"Note: Full pipeline requires trained model. Error: {e}")
        print("\nTo use: python gec_pipeline.py --model_path <path> --mode interactive")