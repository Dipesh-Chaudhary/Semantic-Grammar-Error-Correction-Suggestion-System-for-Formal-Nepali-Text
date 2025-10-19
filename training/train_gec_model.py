"""
Training script for Nepali GEC Model with QLoRA
Optimized for Lightning AI free tier (L4 GPU, 22 hours/month)
"""
import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from typing import Dict, List
import wandb

import sys
sys.path.append('..')
from config import Config


class NepaliGECTrainer:
    """Train Nepali GEC model with QLoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_config = config.model
        self.training_config = config.training
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        
    def setup_model(self):
        """Initialize model with QLoRA configuration"""
        print("=" * 60)
        print("SETTING UP MODEL WITH QLORA")
        print("=" * 60)
        
        # Load tokenizer
        print(f"\nLoading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_fast=True
        )
        
        # Configure 4-bit quantization
        if self.model_config.use_qlora:
            print("\nConfiguring 4-bit quantization (QLoRA)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.model_config.load_in_4bit,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # Load base model
        print(f"\nLoading base model: {self.model_config.model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config if self.model_config.use_qlora else None,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.training_config.bf16 else torch.float16
        )
        
        # Prepare model for k-bit training
        if self.model_config.use_qlora:
            print("\nPreparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        print("\nConfiguring LoRA...")
        lora_config = LoraConfig(
            r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=self.model_config.lora_target_modules,
            lora_dropout=self.model_config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        print("\nModel setup complete!")
        return self.model, self.tokenizer
    
    def load_and_preprocess_data(self, train_file: str, dev_file: str) -> Dict:
        """Load and preprocess training data"""
        print("\n" + "=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Load datasets
        print(f"\nLoading train data from: {train_file}")
        print(f"Loading dev data from: {dev_file}")
        
        data_files = {
            "train": train_file,
            "validation": dev_file
        }
        
        dataset = load_dataset('json', data_files=data_files)
        
        print(f"\nTrain samples: {len(dataset['train'])}")
        print(f"Dev samples: {len(dataset['validation'])}")
        
        # Preprocess function
        def preprocess_function(examples):
            # Add task prefix to inputs
            inputs = [
                self.model_config.task_prefix + source 
                for source in examples['source']
            ]
            targets = examples['target']
            
            # Tokenize
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.model_config.max_length,
                truncation=True,
                padding='max_length'
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.model_config.max_length,
                    truncation=True,
                    padding='max_length'
                )
            
            model_inputs['labels'] = labels['input_ids']
            
            return model_inputs
        
        # Apply preprocessing
        print("\nPreprocessing datasets...")
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        print("Preprocessing complete!")
        return tokenized_datasets
    
    def setup_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments"""
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_steps=self.training_config.warmup_steps,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim=self.training_config.optim,
            logging_dir=f"{self.training_config.output_dir}/logs",
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            report_to=self.training_config.report_to,
            seed=self.training_config.seed,
            predict_with_generate=True,
            generation_max_length=self.model_config.max_length,
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        
        # Decode labels (replace -100 with pad token)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True
        )
        
        # Compute exact match
        exact_match = sum([
            pred.strip() == label.strip() 
            for pred, label in zip(decoded_preds, decoded_labels)
        ]) / len(decoded_preds)
        
        return {
            "exact_match": exact_match
        }
    
    def train(self, train_file: str, dev_file: str):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        # Setup model
        self.setup_model()
        
        # Load and preprocess data
        tokenized_datasets = self.load_and_preprocess_data(train_file, dev_file)
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer, eval_metrics


def main():
    """Main training execution"""
    # Initialize config
    config = Config()
    
    # Initialize W&B (optional)
    if config.training.report_to == "wandb":
        wandb.init(
            project="nepali-semantic-gec",
            config=config.to_dict(),
            name=f"mt5-small-qlora-r{config.model.lora_r}"
        )
    
    # Initialize trainer
    trainer = NepaliGECTrainer(config)
    
    # Train model
    model, metrics = trainer.train(
        train_file=config.data.train_output_path,
        dev_file=config.data.dev_output_path
    )
    
    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))
    
    if config.training.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()