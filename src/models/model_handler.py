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
            tuple: Loaded model and tokenizer.
        """
        try:
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

    def save_model(self, model, tokenizer, path="fine_tuned_model", save_method="lora"):
        """
        Saves the model and tokenizer to the specified path.

        Args:
            model: The trained model to be saved.
            tokenizer: The tokenizer to be saved.
            path (str): Directory path where the model and tokenizer will be saved.
            save_method (str): Method to save the model. Options include "lora", "merged_16bit", "merged_4bit", "merged_lora".
        """
        try:
            if save_method == "lora": 
                model.save_pretrained(path) # This only saves the LoRA adapters, and not the full model
                tokenizer.save_pretrained(path)
            elif save_method == "merged_16bit":
                model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit") # Merge to 16bit
            elif save_method == "merged_4bit":
                model.save_pretrained_merged(path, tokenizer, save_method="merged_4bit") # Merge to 4bit
            elif save_method == "merged_lora":
                model.save_pretrained_merged(path, tokenizer, save_method="lora") # Just Lora adapters
            # Save to GGUF
            # elif save_method == "gguf":
                # model.save_pretrained_gguf(path, tokenizer, quantization_method="f16") # Save to 16bit GGUF --or--
                # model.save_pretrained_gguf(path, tokenizer, quantization_method="q4_k_m") # Save to q4_k_m GGUF --or--
                # model.save_pretrained_gguf(path, tokenizer) # Save to 8bit Q8_0
            else:
                raise ValueError(f"Unsupported save method: {save_method}")

            print(f"Model and tokenizer saved successfully at {path}")

        except Exception as e:
            print(f"Error saving model or tokenizer: {e}")
            raise
