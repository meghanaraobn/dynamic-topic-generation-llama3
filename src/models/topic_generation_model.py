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
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.dataset_name = "ankitagr01/dynamic_topic_modeling_arxiv_abstracts"
        self.pre_trained_model_path = "unsloth/llama-3-8b-bnb-4bit"
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
                per_device_train_batch_size = self.args.per_device_train_batch_size,
                gradient_accumulation_steps = self.args.gradient_accumulation_steps,
                warmup_steps = self.args.warmup_steps,
                max_steps = self.args.max_steps,
                learning_rate = self.args.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = self.args.logging_steps,
                evaluation_strategy = self.args.evaluation_strategy,
                eval_steps = self.args.eval_steps,
                save_steps = self.args.save_steps,
                optim = self.args.optim,
                weight_decay = self.args.weight_decay,
                lr_scheduler_type = self.args.lr_scheduler_type,
                seed = self.args.seed,
                output_dir = self.args.output_dir,
                report_to = "wandb" if self.args.wandb_key else None,
                logging_dir = self.args.logging_dir,
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
            self.model_handler.save_model(self.model, self.tokenizer, self.args.model_save_path, self.args.model_save_method)
        except Exception as e:
            print(f"Error occurred during training: {str(e)}")
            raise
        finally:
            # Finish WandB run
            if self.run:
                self.run.finish()