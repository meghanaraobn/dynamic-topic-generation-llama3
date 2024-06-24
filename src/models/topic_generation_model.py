from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
from unsloth import is_bfloat16_supported
from models.model_handler import ModelHandler
from scripts.data_format import DataFormat

class TopicGenerationModel:
    """
    Class to handle training of dynamic topic generation model using SFTTrainer.
    """

    def __init__(self, args):
        """
        Initialize the model with dataset name, pre-trained model path and training arguments.

        Args:
            args (argparse): Parsed command-line arguments.
        """
        self.args = args
        self.dataset_name = "ankitagr01/dynamic_topic_modeling_arxiv_abstracts"
        self.pre_trained_model_path = "unsloth/llama-3-8b-bnb-4bit"
        self.max_seq_length = 2048
        self.model_handler = ModelHandler()
        try:
            self.model, self.tokenizer = self.model_handler.load_model(self.pre_trained_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        self.run = None  # wandb run

    def load_datasets(self):
        """
        Load and prepare dataset.

        Returns:
            dataset, eval_dataset (tuple): A tuple containing the training dataset and evaluation dataset.
        """
        dataset = load_dataset(self.dataset_name, split="train")
        eval_dataset = load_dataset(self.dataset_name, split="test")
        
        data_format = DataFormat(self.tokenizer.eos_token)
        dataset = dataset.map(data_format.format_prompts, batched=True)
        eval_dataset = eval_dataset.map(data_format.format_prompts, batched=True)

        return dataset, eval_dataset

    def setup_trainer(self, dataset, eval_dataset):
        """
        Setup SFTTrainer with training arguments.

        Args:
            dataset (tuple): The training dataset.
            eval_dataset (tuple): The evaluation dataset.

        Returns:
            trainer (SFTTrainer): An instance of SFTTrainer configured with the provided datasets and training arguments.
        """
        trainer = SFTTrainer(
            model = self.model,  # The pre-trained model to be fine-tuned
            tokenizer = self.tokenizer,  # The tokenizer associated with the model
            train_dataset = dataset,  # Dataset used for training
            eval_dataset = eval_dataset,  # Dataset used for evaluation
            dataset_text_field = "text",  # Only take 'text' part from the dataset
            max_seq_length = self.max_seq_length,  # Maximum sequence length for input text
            dataset_num_proc = 2,  # Number of processes for dataset processing
            packing = False,  # When set to True, it makes training faster by combining multiple short sequences into a single long sequence.
            args=TrainingArguments(
                per_device_train_batch_size = self.args.per_device_train_batch_size,  # Batch size for training
                gradient_accumulation_steps = self.args.gradient_accumulation_steps,  # Number of steps to accumulate gradients 
                warmup_steps = self.args.warmup_steps,  # Number of warmup steps for learning rate scheduler
                num_train_epochs = self.args.num_train_epochs, # Number of epochs for training
                #max_steps = self.args.max_steps,  # Maximum number of training steps
                learning_rate = self.args.learning_rate, # Initial learning rate
                fp16 = not is_bfloat16_supported(), # Use 16-bit precision if bfloat16 is not supported
                bf16 = is_bfloat16_supported(),  # Use bfloat16 precision if supported
                logging_steps = self.args.logging_steps,  # Logging steps interval
                eval_strategy = self.args.evaluation_strategy,  # Evaluation step interval
                eval_steps = self.args.eval_steps,  # Evaluation step interval
                save_steps = self.args.save_steps,  # Model save step interval
                optim = self.args.optim,  # AdamW optimizer in 8-bit precision to reduce memory usage.
                weight_decay = self.args.weight_decay,  # Weight decay for the optimizer
                lr_scheduler_type = self.args.lr_scheduler_type,  # Learning rate scheduler
                seed = self.args.seed,  # Random seed for reproducibility
                output_dir = self.args.output_dir,  # Directory to save model checkpoints
                report_to = "wandb" if self.args.wandb_key else None,  # Reporting tool for logging in wandb
                logging_dir = self.args.logging_dir,  # Directory for logging
                run_name = self.args.project_name,
            ),
        )
        return trainer

    def train(self):
        """
        Method to perform model training using SFTTrainer.
        """
        try:
            if self.args.wandb_key:
                # Initialize WandB
                wandb.login(key=self.args.wandb_key)
                self.run = wandb.init(project=self.args.project_name)
            else:
                wandb.login(relogin=True)
            
            # Load datasets
            dataset, eval_dataset = self.load_datasets()

            # Model Train
            trainer = self.setup_trainer(dataset, eval_dataset)
            print("Model training started")
            trainer.train()

            # Model Save
            self.model_handler.save_model(self.model, self.tokenizer, self.args.model_save_path)
        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
            raise
        finally:
            # Finish WandB run
            if self.run:
                self.run.finish()