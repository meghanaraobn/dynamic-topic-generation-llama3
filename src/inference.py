import argparse
from models.dynamic_topic_generation_model import TopicGenerationModel
from scripts.data_format import DataFormat

class ModelInference:
    """
    Class for performing topic generation inference using a pre-trained model.
    """

    def __init__(self, model_path):
        """
        Initialize the model with the given model path.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.topic_generation_model = TopicGenerationModel()
        try:
            self.model, self.tokenizer = self.topic_generation_model.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def extract_response_text(self, output_text):
        """
        Extract the response text from the output text.

        Args:
            output_text (str): The complete output text.

        Returns:
            str: The extracted response text.
        """
        response_marker = "### Response:"
        start_index = output_text.find(response_marker)
        if start_index == -1:
            return "" 
        
        start_index += len(response_marker)
        
        # Extract only the response text following the '### Response:', excluding any subsequent text.
        end_index = output_text.find("###", start_index)
        if end_index == -1:
            end_index = len(output_text)
        
        response_text = output_text[start_index:end_index].strip()
        return response_text
    
    def generate_topic(self, input_text):
        """
        Generate a topic for the given input text.

        Args:
            input_text (str): The input text for which to generate a topic.

        Returns:
            str: The generated topic.
        """
        try:
            dataFormat = DataFormat(self.tokenizer.eos_token)
            formatted_input = dataFormat.format_single_prompt(input_text)
            input_ids = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
            output = self.model.generate(**input_ids, max_new_tokens=8, use_cache=True)
            response = self.extract_response_text(self.tokenizer.batch_decode(output, skip_special_tokens=True)[0])
            return response
        except Exception as e:
            print(f"Error during topic generation: {e}")
            raise

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Topic Generation Inference")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for topic generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_inference = ModelInference(args.model_path)
    topic = model_inference.generate_topic(args.input_text)
    print('======================================================================================')
    print(f"Generated Topic: {topic}")
    print('======================================================================================')