"""
Utility functions for Nepali GEC system
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata
import re


class NepaliTextProcessor:
    """Utilities for Nepali text preprocessing"""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Nepali Unicode text
        Handles NFC/NFD normalization issues
        """
        # Normalize to NFC (canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        return normalized
    
    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Remove extra whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text
    
    @staticmethod
    def is_nepali_text(text: str, min_ratio: float = 0.7) -> bool:
        """
        Check if text is primarily Nepali
        
        Args:
            text: Input text
            min_ratio: Minimum ratio of Devanagari characters
        """
        if not text:
            return False
        
        devanagari_chars = sum(
            1 for char in text 
            if '\u0900' <= char <= '\u097F'
        )
        
        total_chars = sum(1 for char in text if not char.isspace())
        
        if total_chars == 0:
            return False
        
        return (devanagari_chars / total_chars) >= min_ratio
    
    @staticmethod
    def clean_corpus_line(line: str) -> str:
        """
        Clean a line from corpus
        - Normalize Unicode
        - Remove extra spaces
        - Remove special characters (keep Devanagari and punctuation)
        """
        # Normalize
        line = NepaliTextProcessor.normalize_unicode(line)
        
        # Remove URLs
        line = re.sub(r'http\S+|www\S+', '', line)
        
        # Remove emails
        line = re.sub(r'\S+@\S+', '', line)
        
        # Remove excess punctuation
        line = re.sub(r'[।॥]{2,}', '।', line)
        
        # Remove extra spaces
        line = NepaliTextProcessor.remove_extra_spaces(line)
        
        return line


class DatasetUtils:
    """Utilities for dataset management"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        """Save data to JSONL file"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    @staticmethod
    def split_dataset(
        data: List[Dict],
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/dev/test"""
        import random
        
        random.seed(seed)
        random.shuffle(data)
        
        total = len(data)
        train_end = int(total * train_ratio)
        dev_end = int(total * (train_ratio + dev_ratio))
        
        train = data[:train_end]
        dev = data[train_end:dev_end]
        test = data[dev_end:]
        
        return train, dev, test
    
    @staticmethod
    def merge_datasets(datasets: List[List[Dict]]) -> List[Dict]:
        """Merge multiple datasets"""
        merged = []
        for dataset in datasets:
            merged.extend(dataset)
        return merged
    
    @staticmethod
    def filter_by_error_type(
        data: List[Dict],
        error_types: List[str]
    ) -> List[Dict]:
        """Filter dataset by error types"""
        filtered = []
        for item in data:
            if any(et in item['error_types'] for et in error_types):
                filtered.append(item)
        return filtered
    
    @staticmethod
    def get_statistics(data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        from collections import Counter
        
        total = len(data)
        error_counter = Counter()
        semantic_count = 0
        multi_error_count = 0
        
        for item in data:
            error_counter.update(item['error_types'])
            if item.get('is_semantic', False):
                semantic_count += 1
            if len(item['error_types']) > 1:
                multi_error_count += 1
        
        return {
            'total_samples': total,
            'semantic_samples': semantic_count,
            'multi_error_samples': multi_error_count,
            'error_distribution': dict(error_counter),
            'avg_errors_per_sample': sum(error_counter.values()) / total
        }


class ModelUtils:
    """Utilities for model management"""
    
    @staticmethod
    def count_parameters(model) -> Dict[str, int]:
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percentage': (trainable / total * 100) if total > 0 else 0
        }
    
    @staticmethod
    def get_model_size_mb(model) -> float:
        """Get model size in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    @staticmethod
    def save_training_config(config: Dict, output_dir: str):
        """Save training configuration"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        config_file = Path(output_dir) / 'training_config.json'
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class EvaluationUtils:
    """Utilities for evaluation"""
    
    @staticmethod
    def compute_accuracy_by_length(
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """Compute accuracy stratified by sentence length"""
        from collections import defaultdict
        
        length_bins = [(0, 5), (6, 10), (11, 20), (21, 100)]
        accuracy_by_length = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            length = len(ref.split())
            is_correct = pred.strip() == ref.strip()
            
            for min_len, max_len in length_bins:
                if min_len <= length <= max_len:
                    accuracy_by_length[f"{min_len}-{max_len}"].append(int(is_correct))
                    break
        
        result = {}
        for bin_name, correct_list in accuracy_by_length.items():
            if correct_list:
                result[bin_name] = {
                    'accuracy': sum(correct_list) / len(correct_list) * 100,
                    'count': len(correct_list)
                }
        
        return result
    
    @staticmethod
    def create_error_analysis_report(
        sources: List[str],
        predictions: List[str],
        references: List[str],
        error_types: List[List[str]]
    ) -> str:
        """Create detailed error analysis report"""
        report = []
        report.append("=" * 80)
        report.append("ERROR ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall statistics
        total = len(predictions)
        correct = sum(p == r for p, r in zip(predictions, references))
        
        report.append(f"\nOverall Statistics:")
        report.append(f"  Total samples: {total}")
        report.append(f"  Correct: {correct} ({correct/total*100:.2f}%)")
        report.append(f"  Incorrect: {total - correct} ({(total-correct)/total*100:.2f}%)")
        
        # Examples of failures
        report.append(f"\nFailure Examples (first 10):")
        failure_count = 0
        for src, pred, ref, etypes in zip(sources, predictions, references, error_types):
            if pred != ref and failure_count < 10:
                report.append(f"\n  Source:     {src}")
                report.append(f"  Prediction: {pred}")
                report.append(f"  Reference:  {ref}")
                report.append(f"  Error types: {', '.join(etypes)}")
                failure_count += 1
        
        return '\n'.join(report)


class FileUtils:
    """File and path utilities"""
    
    @staticmethod
    def ensure_dir(path: str):
        """Ensure directory exists"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    @staticmethod
    def list_checkpoints(output_dir: str) -> List[str]:
        """List all checkpoints in output directory"""
        checkpoint_dirs = []
        output_path = Path(output_dir)
        
        if output_path.exists():
            for item in output_path.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint-'):
                    checkpoint_dirs.append(str(item))
        
        return sorted(checkpoint_dirs)
    
    @staticmethod
    def clean_checkpoints(output_dir: str, keep_best: int = 3):
        """Clean old checkpoints, keep only best N"""
        checkpoints = FileUtils.list_checkpoints(output_dir)
        
        if len(checkpoints) > keep_best:
            # Keep most recent checkpoints
            to_remove = checkpoints[:-keep_best]
            
            for checkpoint in to_remove:
                import shutil
                shutil.rmtree(checkpoint)
                print(f"Removed checkpoint: {checkpoint}")


# Convenience functions
def quick_load_data(file_path: str) -> List[Dict]:
    """Quick load JSONL data"""
    return DatasetUtils.load_jsonl(file_path)


def quick_save_data(data: List[Dict], file_path: str):
    """Quick save JSONL data"""
    DatasetUtils.save_jsonl(data, file_path)


def clean_nepali_text(text: str) -> str:
    """Quick clean Nepali text"""
    processor = NepaliTextProcessor()
    return processor.clean_corpus_line(text)


# Example usage
if __name__ == "__main__":
    print("Nepali GEC Utilities")
    print("=" * 60)
    
    # Test text processing
    processor = NepaliTextProcessor()
    
    test_text = "  म  किताब   पढ्छु  ।  "
    cleaned = processor.clean_corpus_line(test_text)
    print(f"\nOriginal: '{test_text}'")
    print(f"Cleaned:  '{cleaned}'")
    
    # Test Nepali detection
    is_nepali = processor.is_nepali_text("म नेपाली भाषा बोल्छु")
    print(f"\nIs Nepali: {is_nepali}")
    
    # Test dataset utilities
    print("\nDataset Utilities:")
    sample_data = [
        {'source': 'error1', 'target': 'correct1', 'error_types': ['TYPE_A']},
        {'source': 'error2', 'target': 'correct2', 'error_types': ['TYPE_B', 'TYPE_C']},
    ]
    
    stats = DatasetUtils.get_statistics(sample_data)
    print(f"Statistics: {stats}")
    
    print("\n✓ Utilities working correctly!")