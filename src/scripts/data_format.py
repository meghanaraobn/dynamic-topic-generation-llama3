
class DataFormat:
    """
    Class to handle formatting prompts for each example in a dataset.
    """

    def __init__(self, eos_token):
        """
        Initialize DataFormat with necessary attributes.

        Args:
            eos_token (str): End-of-sequence token to append to formatted prompts.
        """
        self.input_key = "Abstract"  # "Abstract" is the key for abstracts in dataset
        self.response_key = "Topic"  # "Topic" is the key for topics in dataset
        self.instruction = "Please generate a meaningful topic for the following article."
        self.eos_token = eos_token
        self.prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def format_prompts(self, dataset):
        """
        Format prompts for each sample in the dataset.

        Args:
            dataset (dict): Dictionary containing samples with input and response data.

        Returns:
            dict: Dictionary with formatted text prompts under the key 'text'.
        """
        texts = []
        for inp, resp in zip(dataset[self.input_key], dataset[self.response_key]):
            text = self.prompt.format(self.instruction, inp, resp) + self.eos_token # Add EOS_TOKEN to prevent infinite generation
            texts.append(text)
        return {"text": texts}
    

    def format_single_prompt(self, sample):
        """
        Format prompt for a single sample.

        Args:
            sample (str): Single sample with input text.

        Returns:
            list: List containing the formatted text prompt.
        """
        text = self.prompt.format(self.instruction, sample, "")
        return [text]
    
