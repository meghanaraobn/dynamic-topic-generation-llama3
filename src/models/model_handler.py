from unsloth import FastLanguageModel

class ModelHandler:
    """
    Class for loading and saving a model.
    """
    def __init__(self):
        """
        Initialize the default parameters.
        """
        self.max_seq_length = 2048
        self.load_in_4bit = True
        self.dtype = None

    def load_model(self, model_path):
        """
        Loads the model and tokenizer from the specified model path.

        Args:
            model_path (str): Path to the model.

        Returns:
            model, tokenizer (tuple): Loaded model and tokenizer.
        """
        try:
            # Load model and tokenizer
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = self.max_seq_length,
                dtype = self.dtype,
                load_in_4bit = self.load_in_4bit,
            )
        
            # LoRA adapters are added and so only 1 to 10% of all parameters are updated
            model = FastLanguageModel.get_peft_model(
                model,
                r = 16,  # Any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha = 16,
                lora_dropout = 0,  # Any value, but = 0 is optimized
                bias = "none",  # Any value, but = "none" is optimized
                use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # Rank stabilized LoRA
                loftq_config = None,  # LoftQ
            )

            return model, tokenizer

        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

    def save_model(self, model, tokenizer, path="fine_tuned_model"):
        """
        Saves the model and tokenizer to the specified path.

        Args:
            model (tuple): The trained model to be saved.
            tokenizer (tuple): The tokenizer to be saved.
            path (str): Directory path where the model and tokenizer will be saved.
        """
        try: 
            model.save_pretrained(path) # This only saves the LoRA adapters, and not the full model
            tokenizer.save_pretrained(path)
            print(f"Model and tokenizer saved successfully at {path}")

        except Exception as e:
            print(f"Error saving model or tokenizer: {e}")
            raise
