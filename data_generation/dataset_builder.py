"""
Complete Dataset Builder for Nepali GEC
Combines rule-based, semantic, and synthetic approaches
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
from collections import Counter

# Import error generators
import sys
sys.path.append('..')
from config import Config, ErrorTaxonomy


class NepaliGECDatasetBuilder:
    """Build comprehensive Nepali GEC dataset"""
    
    def __init__(self, config: Config):
        self.config = config
        self.error_taxonomy = ErrorTaxonomy()
        
        # Initialize generators - UPDATED to use advanced semantic generator
        from rule_based_errors import NepaliErrorGenerator
        from advanced_semantic_errors import AdvancedSemanticErrorGenerator
        
        self.rule_generator = NepaliErrorGenerator(seed=config.training.seed)
        self.semantic_generator = AdvancedSemanticErrorGenerator(seed=config.training.seed)
        
        self.dataset = []
        
    def load_clean_corpus(self, corpus_path: str, max_sentences: int = 100000) -> List[str]:
        """
        Load clean Nepali sentences from corpus
        Filters for quality and appropriate length
        """
        print(f"Loading corpus from {corpus_path}...")
        sentences = []
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Filter by length
                    words = line.split()
                    if (self.config.data.min_sentence_length <= len(words) <= 
                        self.config.data.max_sentence_length):
                        
                        # Basic quality checks
                        if self._is_quality_sentence(line):
                            sentences.append(line)
                            
                            if len(sentences) >= max_sentences:
                                break
        except FileNotFoundError:
            print(f"Warning: Corpus file not found at {corpus_path}")
            print("Using example sentences instead...")
            sentences = self._get_example_sentences()
        
        print(f"Loaded {len(sentences)} clean sentences")
        return sentences
    
    def _is_quality_sentence(self, sentence: str) -> bool:
        """Check if sentence meets quality criteria"""
        # Must contain Devanagari characters
        has_devanagari = any('\u0900' <= char <= '\u097F' for char in sentence)
        
        # Not too many numbers or special characters
        special_count = sum(1 for char in sentence if not (char.isalnum() or char.isspace()))
        special_ratio = special_count / len(sentence) if len(sentence) > 0 else 1
        
        return has_devanagari and special_ratio < 0.3
    
    def _get_example_sentences(self) -> List[str]:
        """Fallback example sentences for testing"""
        return [
            "म नेपाली भाषा सिक्दैछु",
            "उनी विद्यालयमा पढ्छन्",
            "हामी काठमाडौंमा बस्छौं",
            "तपाईं कहाँ जानुहुन्छ",
            "बच्चाले खाना खान्छ",
            "शिक्षकले विद्यार्थीलाई पढाउँछन्",
            "गाईले दूध दिन्छ",
            "म किताब पढ्छु",
            "उनीहरू बजार जान्छन्",
            "आमाले खाना पकाउँछिन्",
        ] * 100  # Repeat for testing
    
    def generate_rule_based_dataset(self, clean_sentences: List[str], num_samples: int) -> List[Dict]:
        """Generate rule-based grammatical errors"""
        print(f"\nGenerating {num_samples} rule-based error samples...")
        
        samples = []
        error_type_distribution = Counter()
        
        for _ in tqdm(range(num_samples)):
            sent = random.choice(clean_sentences)
            
            # Decide: single or multi-error (70% single, 30% multi)
            if random.random() < 0.7:
                error_sent, error_type = self.rule_generator.generate_error(sent)
                error_types = [error_type] if error_type != "NONE" else []
            else:
                num_errors = random.randint(2, 3)
                error_sent, error_types = self.rule_generator.generate_multi_error(sent, num_errors)
            
            if error_types:  # Only add if error was generated
                samples.append({
                    'source': error_sent,
                    'target': sent,
                    'error_types': error_types,
                    'is_semantic': False,
                    'generation_method': 'rule_based'
                })
                error_type_distribution.update(error_types)
        
        print(f"Generated {len(samples)} samples")
        print("Error type distribution:", dict(error_type_distribution.most_common()))
        
        return samples
    
    def generate_semantic_dataset(self, clean_sentences: List[str], num_samples: int) -> List[Dict]:
        """Generate semantic errors"""
        print(f"\nGenerating {num_samples} semantic error samples...")
        
        samples = []
        error_type_distribution = Counter()
        
        for _ in tqdm(range(num_samples)):
            sent = random.choice(clean_sentences)
            
            error_sent, error_type = self.semantic_generator.generate_semantic_error_sentence(sent)
            
            if error_type != "NONE":
                samples.append({
                    'source': error_sent,
                    'target': sent,
                    'error_types': [error_type],
                    'is_semantic': True,
                    'generation_method': 'semantic_rules'
                })
                error_type_distribution[error_type] += 1
        
        print(f"Generated {len(samples)} samples")
        print("Error type distribution:", dict(error_type_distribution))
        
        return samples
    
    def generate_combined_errors(self, clean_sentences: List[str], num_samples: int) -> List[Dict]:
        """Generate sentences with both grammatical and semantic errors"""
        print(f"\nGenerating {num_samples} combined error samples...")
        
        samples = []
        
        for _ in tqdm(range(num_samples)):
            sent = random.choice(clean_sentences)
            
            # First add semantic error
            semantic_sent, sem_type = self.semantic_generator.generate_semantic_error_sentence(sent)
            
            # Then add grammatical error
            if sem_type != "NONE":
                gram_sent, gram_type = self.rule_generator.generate_error(semantic_sent)
                
                if gram_type != "NONE":
                    samples.append({
                        'source': gram_sent,
                        'target': sent,
                        'error_types': [sem_type, gram_type],
                        'is_semantic': True,
                        'generation_method': 'combined'
                    })
        
        print(f"Generated {len(samples)} combined samples")
        return samples
    
    def build_complete_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Build complete dataset with train/dev/test splits
        """
        print("=" * 60)
        print("BUILDING COMPLETE NEPALI GEC DATASET")
        print("=" * 60)
        
        # Load clean corpus
        clean_sentences = self.load_clean_corpus(
            self.config.data.raw_corpus_path,
            max_sentences=100000
        )
        
        # Generate different types of errors
        rule_samples = self.generate_rule_based_dataset(
            clean_sentences,
            self.config.data.num_rule_based_samples
        )
        
        semantic_samples = self.generate_semantic_dataset(
            clean_sentences,
            self.config.data.num_semantic_samples
        )
        
        combined_samples = self.generate_combined_errors(
            clean_sentences,
            int(self.config.data.num_rule_based_samples * 0.2)  # 20% combined
        )
        
        # Combine all samples
        all_samples = rule_samples + semantic_samples + combined_samples
        random.shuffle(all_samples)
        
        print(f"\nTotal samples generated: {len(all_samples)}")
        
        # Split into train/dev/test
        train_size = int(len(all_samples) * self.config.data.train_ratio)
        dev_size = int(len(all_samples) * self.config.data.dev_ratio)
        
        train_data = all_samples[:train_size]
        dev_data = all_samples[train_size:train_size + dev_size]
        test_data = all_samples[train_size + dev_size:]
        
        print(f"Train: {len(train_data)}")
        print(f"Dev:   {len(dev_data)}")
        print(f"Test:  {len(test_data)}")
        
        return train_data, dev_data, test_data
    
    def save_dataset(self, train_data: List[Dict], dev_data: List[Dict], 
                     test_data: List[Dict], output_dir: str = "./data/processed"):
        """Save dataset to JSONL files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        splits = {
            'train': train_data,
            'dev': dev_data,
            'test': test_data
        }
        
        for split_name, data in splits.items():
            output_file = output_path / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"Saved {split_name} to {output_file}")
        
        # Save statistics
        self._save_statistics(train_data, dev_data, test_data, output_path)
    
    def _save_statistics(self, train_data, dev_data, test_data, output_path):
        """Save dataset statistics"""
        stats = {
            'total_samples': len(train_data) + len(dev_data) + len(test_data),
            'train_samples': len(train_data),
            'dev_samples': len(dev_data),
            'test_samples': len(test_data),
            'error_type_distribution': {},
            'semantic_samples': 0,
            'multi_error_samples': 0
        }
        
        all_data = train_data + dev_data + test_data
        error_counter = Counter()
        
        for item in all_data:
            error_counter.update(item['error_types'])
            if item['is_semantic']:
                stats['semantic_samples'] += 1
            if len(item['error_types']) > 1:
                stats['multi_error_samples'] += 1
        
        stats['error_type_distribution'] = dict(error_counter)
        
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"\nStatistics saved to {stats_file}")


def main():
    """Main execution"""
    # Initialize config
    config = Config()
    
    # Build dataset
    builder = NepaliGECDatasetBuilder(config)
    train_data, dev_data, test_data = builder.build_complete_dataset()
    
    # Save dataset
    builder.save_dataset(train_data, dev_data, test_data)
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()