import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import wandb
from models.dynamic_topic_generation_model import TopicGenerationModel
from scripts.data_format import DataFormat
from unsloth import is_bfloat16_supported

class ModelConfig:
    """
    Class to encapsulate model configuration settings.
    """
    def __init__(self, args):
        self.use_wandb = args.use_wandb
        self.project_name = args.project_name
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.learning_rate = args.learning_rate
        self.logging_steps = args.logging_steps
        self.evaluation_strategy = args.evaluation_strategy
        self.eval_steps = args.eval_steps
        self.save_steps = args.save_steps
        self.optim = args.optim
        self.weight_decay = args.weight_decay
        self.lr_scheduler_type = args.lr_scheduler_type
        self.seed = args.seed
        self.model_save_path = args.model_save_path
        self.model_save_method = args.model_save_method
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir

class ModelTrain:
    """
    Class to handle training of dynamic topic generation model using SFTTrainer.
    """

    def __init__(self, config):
        """
        Initialize the model with project name and training arguments.

        Args:
            config (ModelConfig): Configuration object containing training parameters.
        """
        self.config = config
        self.dataset_name = "ankitagr01/dynamic_topic_modeling_arxiv_abstracts"
        self.pre_trained_model_path = "unsloth/llama-3-8b-bnb-4bit"
        self.topic_generation_model = TopicGenerationModel()
        try:
            self.model, self.tokenizer = self.topic_generation_model.load_model(self.pre_trained_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        self.run = None  # wandb run

    def load_datasets(self):
        """
        Load and prepare dataset.
        """
        dataset = load_dataset(self.dataset_name, split="train")
        eval_dataset = load_dataset(self.dataset_name, split="test")
        
        data_formatter = DataFormat(self.tokenizer.eos_token)
        dataset = dataset.map(data_formatter.format_prompts, batched=True)
        eval_dataset = eval_dataset.map(data_formatter.format_prompts, batched=True)

        return dataset, eval_dataset

    def setup_trainer(self, dataset, eval_dataset):
        """
        Setup SFTTrainer with training arguments.
        """
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            eval_dataset = eval_dataset,
            dataset_text_field = "text",
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = False,
            args=TrainingArguments(
                per_device_train_batch_size = self.config.per_device_train_batch_size,
                gradient_accumulation_steps = self.config.gradient_accumulation_steps,
                warmup_steps = self.config.warmup_steps,
                max_steps = self.config.max_steps,
                learning_rate = self.config.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.config.logging_steps,
                evaluation_strategy = self.config.evaluation_strategy,
                eval_steps = self.config.eval_steps,
                save_steps = self.config.save_steps,
                optim = self.config.optim,
                weight_decay = self.config.weight_decay,
                lr_scheduler_type = self.config.lr_scheduler_type,
                seed = self.config.seed,
                output_dir = self.config.output_dir,
                report_to = "wandb" if self.config.use_wandb else None,
                logging_dir = self.config.logging_dir
            ),
        )
        return trainer

    def train(self):
        """
        Method to perform model training using SFTTrainer.
        """
        try:
            if self.config.use_wandb:
                # Initialize WandB
                wandb.login(relogin=True)
                self.run = wandb.init(project=self.config.project_name)
            
            # Load datasets
            dataset, eval_dataset = self.load_datasets()

            # Model Train
            trainer = self.setup_trainer(dataset, eval_dataset)
            trainer.train()

            # Model Save
            self.topic_generation_model.save_model(self.model, self.tokenizer, self.config.model_save_path, self.config.model_save_method)
        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
            raise
        finally:
            # Finish WandB run
            if self.run:
                self.run.finish()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training script for dynamic topic generation model")
    parser.add_argument("--use_wandb", action='store_true', help="Enable WandB integration")
    parser.add_argument("--project_name", type=str, default="Llama3_dynamic_topic_generation", help="Name of the project in WandB")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU/device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients before performing optimization")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging training metrics")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch')")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=200, help="Number of steps between saving model checkpoints")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--model_save_path", type=str, default="model", help="Path to save the fine-trained model")
    parser.add_argument("--model_save_method", type=str, default="lora", help="Method to save the fine-tuned model. Options include 'lora', 'merged_16bit', 'merged_4bit', 'gguf'")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output files (checkpoints, logs)")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save logging files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig(args)
    model_train = ModelTrain(config)
    model_train.train()