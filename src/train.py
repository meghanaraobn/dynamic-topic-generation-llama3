import argparse
from models.topic_generation_model import TopicGenerationModel

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        parser (argparse): An object containing the parsed command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser(description="Dynamic topic generation model arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wandb_key", type=str, help="wandb API key")
    parser.add_argument("--project_name", type=str, default="Dynamic_topic_generation_Llama3", help="Name of the project")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients before performing optimization")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs for training")
    #parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer")
    parser.add_argument("--logging_steps", type=int, default=25, help="Number of steps between logging training metrics")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy ('steps' or 'epoch')")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of steps between saving model checkpoints")
    parser.add_argument("--optim", type=str, default="adamw_8bit", help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")
    parser.add_argument("--model_save_path", type=str, default="fine_tuned_model", help="Path to save the fine-trained model")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output files")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save logging files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = TopicGenerationModel(args)
    model.train() # Model training starts
