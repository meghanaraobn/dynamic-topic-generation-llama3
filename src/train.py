import argparse
from models.topic_generation_model import TopicGenerationModel

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training script for dynamic topic generation model")
    parser.add_argument("--wandb_key", type=str, help="Enable WandB integration")
    parser.add_argument("--project_name", type=str, default="Dynamic_topic_generation_Llama3", help="Name of the project in WandB")
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
    parser.add_argument("--model_save_path", type=str, default="fine_tuned_model", help="Path to save the fine-trained model")
    parser.add_argument("--model_save_method", type=str, default="lora", help="Method to save the fine-tuned model. Options include 'lora', 'merged_16bit', 'merged_4bit', 'gguf'")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output files (checkpoints, logs)")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save logging files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = TopicGenerationModel(args)
    model.train()
