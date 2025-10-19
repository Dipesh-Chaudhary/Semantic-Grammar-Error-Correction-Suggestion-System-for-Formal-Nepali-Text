"""
Complete End-to-End Pipeline for Nepali Semantic-Aware GEC
Executes all stages: Data Generation -> Training -> Evaluation
"""
import os
import sys
import argparse
import json
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_generation.dataset_builder import NepaliGECDatasetBuilder
from training.train_gec_model import NepaliGECTrainer
from evaluation.evaluate_gec import NepaliGECEvaluator
from inference.gec_pipeline import NepaliGECPipeline


class CompletePipeline:
    """Execute complete GEC pipeline"""
    
    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            # Load custom config
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = Config()
            # Override with custom settings (simplified)
        else:
            self.config = Config()
        
        self.start_time = time.time()
    
    def log_stage(self, stage: str):
        """Log pipeline stage"""
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(f"STAGE: {stage}")
        print(f"Elapsed Time: {elapsed/60:.1f} minutes")
        print("=" * 70 + "\n")
    
    def stage_1_data_generation(self, skip_if_exists: bool = True):
        """Stage 1: Generate training dataset"""
        self.log_stage("1. DATA GENERATION")
        
        # Check if data already exists
        if skip_if_exists and os.path.exists(self.config.data.train_output_path):
            print(f"Data already exists at {self.config.data.train_output_path}")
            print("Skipping data generation (use --regenerate to force)")
            return
        
        # Build dataset
        builder = NepaliGECDatasetBuilder(self.config)
        train_data, dev_data, test_data = builder.build_complete_dataset()
        
        # Save dataset
        builder.save_dataset(train_data, dev_data, test_data)
        
        print(f"\n✓ Dataset generated successfully")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Dev:   {len(dev_data)} samples")
        print(f"  Test:  {len(test_data)} samples")
    
    def stage_2_model_training(self, skip_if_exists: bool = False):
        """Stage 2: Train GEC model"""
        self.log_stage("2. MODEL TRAINING")
        
        # Check if model already exists
        model_path = Path(self.config.training.output_dir)
        if skip_if_exists and model_path.exists() and (model_path / "adapter_model.bin").exists():
            print(f"Model already exists at {model_path}")
            print("Skipping training (use --retrain to force)")
            return
        
        # Initialize trainer
        trainer = NepaliGECTrainer(self.config)
        
        # Train
        model, metrics = trainer.train(
            train_file=self.config.data.train_output_path,
            dev_file=self.config.data.dev_output_path
        )
        
        print(f"\n✓ Model trained successfully")
        print(f"  Output: {self.config.training.output_dir}")
        print(f"  Final loss: {metrics.get('eval_loss', 'N/A')}")
    
    def stage_3_evaluation(self):
        """Stage 3: Evaluate on test set"""
        self.log_stage("3. EVALUATION")
        
        # Load test data
        test_data = []
        with open(self.config.data.test_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        print(f"Loaded {len(test_data)} test samples")
        
        # Initialize pipeline for inference
        try:
            pipeline = NepaliGECPipeline(
                model_path=self.config.training.output_dir,
                base_model=self.config.model.model_name,
                use_semantic_validation=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Skipping evaluation")
            return
        
        # Generate predictions
        print("Generating predictions...")
        sources = [item['source'] for item in test_data]
        targets = [item['target'] for item in test_data]
        error_types = [item['error_types'] for item in test_data]
        
        predictions = pipeline.correct_batch(sources, batch_size=8)
        
        # Evaluate
        evaluator = NepaliGECEvaluator()
        results = evaluator.evaluate_comprehensive(
            predictions=predictions,
            references=targets,
            error_types=error_types,
            compute_bertscore=True
        )
        
        # Print and save results
        evaluator.print_results(results)
        
        results_file = Path(self.config.training.output_dir) / "evaluation_results.json"
        evaluator.save_results(results, str(results_file))
        
        print(f"\n✓ Evaluation complete")
        print(f"  Results saved to: {results_file}")
        
        return results
    
    def stage_4_demo_inference(self):
        """Stage 4: Interactive demo"""
        self.log_stage("4. INTERACTIVE DEMO")
        
        try:
            pipeline = NepaliGECPipeline(
                model_path=self.config.training.output_dir,
                base_model=self.config.model.model_name,
                use_semantic_validation=True
            )
            
            pipeline.interactive_mode()
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure model is trained first")
    
    def run_full_pipeline(
        self,
        skip_data_if_exists: bool = True,
        skip_train_if_exists: bool = False,
        run_demo: bool = False
    ):
        """Run complete pipeline"""
        print("\n" + "=" * 70)
        print(" " * 15 + "NEPALI SEMANTIC-AWARE GEC")
        print(" " * 20 + "COMPLETE PIPELINE")
        print("=" * 70)
        
        try:
            # Stage 1: Data generation
            self.stage_1_data_generation(skip_if_exists=skip_data_if_exists)
            
            # Stage 2: Training
            self.stage_2_model_training(skip_if_exists=skip_train_if_exists)
            
            # Stage 3: Evaluation
            results = self.stage_3_evaluation()
            
            # Stage 4: Demo (optional)
            if run_demo:
                self.stage_4_demo_inference()
            
            # Final summary
            total_time = time.time() - self.start_time
            self.log_stage("PIPELINE COMPLETE")
            print(f"Total execution time: {total_time/60:.1f} minutes")
            
            if results:
                print("\nKey Results:")
                print(f"  Exact Match: {results.get('exact_match', 0):.2f}%")
                print(f"  GLEU:       {results.get('gleu', 0):.2f}")
                print(f"  BLEU:       {results.get('bleu', 0):.2f}")
            
            print("\n✓ All stages completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Nepali Semantic-Aware GEC - Complete Pipeline"
    )
    
    # Pipeline stages
    parser.add_argument('--stage', type=str, 
                       choices=['data', 'train', 'eval', 'demo', 'all'],
                       default='all',
                       help='Pipeline stage to run')
    
    # Options
    parser.add_argument('--config', type=str, default=None,
                       help='Custom config file (JSON)')
    parser.add_argument('--regenerate_data', action='store_true',
                       help='Regenerate data even if exists')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain model even if exists')
    parser.add_argument('--no_demo', action='store_true',
                       help='Skip interactive demo')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline(config_path=args.config)
    
    # Run requested stage(s)
    if args.stage == 'data':
        pipeline.stage_1_data_generation(skip_if_exists=not args.regenerate_data)
    
    elif args.stage == 'train':
        pipeline.stage_2_model_training(skip_if_exists=not args.retrain)
    
    elif args.stage == 'eval':
        pipeline.stage_3_evaluation()
    
    elif args.stage == 'demo':
        pipeline.stage_4_demo_inference()
    
    elif args.stage == 'all':
        pipeline.run_full_pipeline(
            skip_data_if_exists=not args.regenerate_data,
            skip_train_if_exists=not args.retrain,
            run_demo=not args.no_demo
        )


if __name__ == "__main__":
    main()