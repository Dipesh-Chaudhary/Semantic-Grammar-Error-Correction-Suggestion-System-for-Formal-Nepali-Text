"""
Comprehensive Evaluation for Nepali GEC
Implements GLEU, BLEU, BERTScore, and custom metrics
"""
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import Levenshtein

# Import metrics
from sacrebleu import corpus_bleu
from bert_score import score as bert_score
from evaluate import load

import sys
sys.path.append('..')
from config import ErrorTaxonomy


class NepaliGECEvaluator:
    """Comprehensive evaluation for Nepali GEC"""
    
    def __init__(self):
        self.error_taxonomy = ErrorTaxonomy()
        
        # Load metrics
        try:
            self.gleu_metric = load("google_bleu")
        except:
            print("Warning: GLEU metric not available, will compute manually")
            self.gleu_metric = None
    
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute exact match accuracy"""
        assert len(predictions) == len(references)
        
        matches = sum(
            pred.strip() == ref.strip()
            for pred, ref in zip(predictions, references)
        )
        
        return matches / len(predictions) * 100
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]]  # List of lists for multiple references
    ) -> Dict:
        """Compute BLEU score"""
        # sacrebleu expects references as list of lists
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]
        
        # Transpose for sacrebleu format
        refs_transposed = [[ref[i] for ref in references] for i in range(len(references[0]))]
        
        bleu = corpus_bleu(predictions, refs_transposed)
        
        return {
            'bleu': bleu.score,
            'precisions': bleu.precisions,
            'bp': bleu.bp,
            'sys_len': bleu.sys_len,
            'ref_len': bleu.ref_len
        }
    
    def compute_gleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute GLEU (Google BLEU) - sentence-level BLEU
        More suitable for GEC than corpus-level BLEU
        """
        if self.gleu_metric:
            result = self.gleu_metric.compute(
                predictions=predictions,
                references=references
            )
            return result['google_bleu'] * 100
        else:
            # Manual GLEU computation (simplified)
            scores = []
            for pred, ref in zip(predictions, references):
                # Token-level overlap
                pred_tokens = set(pred.split())
                ref_tokens = set(ref.split())
                
                if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                    scores.append(1.0)
                elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    scores.append(0.0)
                else:
                    overlap = len(pred_tokens & ref_tokens)
                    scores.append(overlap / max(len(pred_tokens), len(ref_tokens)))
            
            return np.mean(scores) * 100
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        model_type: str = "bert-base-multilingual-cased"
    ) -> Dict:
        """Compute BERTScore"""
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=model_type,
            lang="ne",
            verbose=False
        )
        
        return {
            'precision': P.mean().item() * 100,
            'recall': R.mean().item() * 100,
            'f1': F1.mean().item() * 100
        }
    
    def compute_edit_distance(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """Compute character-level edit distance metrics"""
        distances = []
        normalized_distances = []
        
        for pred, ref in zip(predictions, references):
            dist = Levenshtein.distance(pred, ref)
            distances.append(dist)
            
            # Normalize by reference length
            max_len = max(len(ref), 1)
            normalized_distances.append(dist / max_len)
        
        return {
            'mean_edit_distance': np.mean(distances),
            'normalized_edit_distance': np.mean(normalized_distances),
            'median_edit_distance': np.median(distances)
        }
    
    def compute_error_type_metrics(
        self,
        predictions: List[str],
        references: List[str],
        error_types: List[List[str]]  # List of error types per sample
    ) -> Dict:
        """
        Compute metrics per error type
        Useful for ablation studies
        """
        error_type_accuracy = defaultdict(list)
        
        for pred, ref, etypes in zip(predictions, references, error_types):
            is_correct = pred.strip() == ref.strip()
            
            for etype in etypes:
                error_type_accuracy[etype].append(int(is_correct))
        
        # Compute accuracy per error type
        metrics = {}
        for etype, correct_list in error_type_accuracy.items():
            metrics[etype] = {
                'accuracy': np.mean(correct_list) * 100,
                'count': len(correct_list)
            }
        
        return metrics
    
    def compute_semantic_metrics(
        self,
        predictions: List[str],
        semantic_labels: List[bool]  # True if semantically correct
    ) -> Dict:
        """
        Compute metrics for semantic correctness
        """
        if not semantic_labels:
            return {}
        
        semantic_accuracy = np.mean(semantic_labels) * 100
        
        return {
            'semantic_accuracy': semantic_accuracy,
            'semantic_errors': len(semantic_labels) - sum(semantic_labels)
        }
    
    def evaluate_comprehensive(
        self,
        predictions: List[str],
        references: List[str],
        error_types: List[List[str]] = None,
        semantic_labels: List[bool] = None,
        compute_bertscore: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation with all metrics
        
        Args:
            predictions: Model predictions
            references: Ground truth corrections
            error_types: List of error types per sample (optional)
            semantic_labels: Semantic correctness labels (optional)
            compute_bertscore: Whether to compute BERTScore (slow)
        
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        results = {}
        
        # Exact match
        print("\nComputing exact match...")
        results['exact_match'] = self.compute_exact_match(predictions, references)
        
        # BLEU
        print("Computing BLEU...")
        bleu_results = self.compute_bleu(predictions, [[ref] for ref in references])
        results['bleu'] = bleu_results['bleu']
        results['bleu_details'] = bleu_results
        
        # GLEU
        print("Computing GLEU...")
        results['gleu'] = self.compute_gleu(predictions, references)
        
        # Edit distance
        print("Computing edit distance...")
        edit_results = self.compute_edit_distance(predictions, references)
        results.update(edit_results)
        
        # BERTScore
        if compute_bertscore:
            print("Computing BERTScore (this may take a while)...")
            bert_results = self.compute_bertscore(predictions, references)
            results['bertscore'] = bert_results
        
        # Error type metrics
        if error_types:
            print("Computing error-type specific metrics...")
            error_type_results = self.compute_error_type_metrics(
                predictions, references, error_types
            )
            results['error_type_metrics'] = error_type_results
        
        # Semantic metrics
        if semantic_labels:
            print("Computing semantic metrics...")
            semantic_results = self.compute_semantic_metrics(
                predictions, semantic_labels
            )
            results.update(semantic_results)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        # Main metrics
        print("\n--- Main Metrics ---")
        print(f"Exact Match:    {results.get('exact_match', 0):.2f}%")
        print(f"BLEU:          {results.get('bleu', 0):.2f}")
        print(f"GLEU:          {results.get('gleu', 0):.2f}")
        
        # Edit distance
        print("\n--- Edit Distance ---")
        print(f"Mean:          {results.get('mean_edit_distance', 0):.2f}")
        print(f"Normalized:    {results.get('normalized_edit_distance', 0):.4f}")
        
        # BERTScore
        if 'bertscore' in results:
            print("\n--- BERTScore ---")
            bs = results['bertscore']
            print(f"Precision:     {bs['precision']:.2f}")
            print(f"Recall:        {bs['recall']:.2f}")
            print(f"F1:           {bs['f1']:.2f}")
        
        # Semantic metrics
        if 'semantic_accuracy' in results:
            print("\n--- Semantic Metrics ---")
            print(f"Accuracy:      {results['semantic_accuracy']:.2f}%")
            print(f"Errors:        {results.get('semantic_errors', 0)}")
        
        # Error type breakdown
        if 'error_type_metrics' in results:
            print("\n--- Error Type Breakdown ---")
            etm = results['error_type_metrics']
            for etype, metrics in sorted(etm.items()):
                print(f"{etype:20s}: {metrics['accuracy']:.1f}% ({metrics['count']} samples)")
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")


def evaluate_from_file(
    predictions_file: str,
    references_file: str,
    output_file: str = None,
    compute_bertscore: bool = True
):
    """
    Evaluate predictions from files
    
    File formats:
    - predictions_file: One prediction per line
    - references_file: One reference per line (or JSONL with 'target' field)
    """
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    # Load references
    references = []
    error_types_list = []
    
    with open(references_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        f.seek(0)
        
        # Check if JSONL format
        if first_line.strip().startswith('{'):
            for line in f:
                item = json.loads(line)
                references.append(item['target'])
                error_types_list.append(item.get('error_types', []))
        else:
            references = [line.strip() for line in f]
    
    # Ensure same length
    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]
    if error_types_list:
        error_types_list = error_types_list[:min_len]
    
    print(f"Loaded {len(predictions)} predictions and {len(references)} references")
    
    # Evaluate
    evaluator = NepaliGECEvaluator()
    results = evaluator.evaluate_comprehensive(
        predictions,
        references,
        error_types=error_types_list if error_types_list else None,
        compute_bertscore=compute_bertscore
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if output_file:
        evaluator.save_results(results, output_file)
    
    return results


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Nepali GEC")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Predictions file (one per line)')
    parser.add_argument('--references', type=str, required=True,
                       help='References file (one per line or JSONL)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--no_bertscore', action='store_true',
                       help='Skip BERTScore computation (faster)')
    
    args = parser.parse_args()
    
    evaluate_from_file(
        args.predictions,
        args.references,
        args.output,
        compute_bertscore=not args.no_bertscore
    )


if __name__ == "__main__":
    # Example usage
    print("Nepali GEC Evaluator - Example Usage\n")
    
    # Simulated data
    predictions = [
        "म किताब पढ्छु",
        "उनी घरमा छन्",
        "तपाईं कहाँ जानुहुन्छ"
    ]
    
    references = [
        "म किताब पढ्छु",
        "उनी घरमा छन्",
        "तपाईं कहाँ जानुहुन्छ"
    ]
    
    error_types = [
        ["ORTH:VOWEL"],
        ["MORPH:VERB_AGR"],
        ["SEM:HONOR"]
    ]
    
    # Evaluate
    evaluator = NepaliGECEvaluator()
    results = evaluator.evaluate_comprehensive(
        predictions,
        references,
        error_types=error_types,
        compute_bertscore=False  # Skip for example
    )
    
    evaluator.print_results(results)