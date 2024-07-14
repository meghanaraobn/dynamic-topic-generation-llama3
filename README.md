# Dynamic Topic Generation using Llama 3
Dynamic topic generation is the process of automatically identifying and categorizing the main themes or subjects within a collection of text data. Unlike traditional methods that rely on predefined categories, dynamic topic generation adapts to the content being analyzed, allowing topics to vary based on the unique characteristics of the texts   .

### Why Llama 3? 
<img src="https://github.com/meghanaraobn/SUMM-AI-TASK/assets/76180138/675b0481-ae76-48bf-8936-6e99c253356c" width="20%" height="15%" align="right" />

[Llama 3](https://ai.meta.com/blog/meta-llama-3/), a Large Language Model (LLM), is chosen for this task due to its strong capabilities in understanding and generating natural language text. Llama 3 can effectively capture the semantic meaning and context from text paragraphs, making it suitable for dynamic topic generation.

### Objective
The objective of this project is to fine-tune Llama 3 to perform dynamic topic generation for a given text paragraph.

## Datasets
The dataset [ankitagr01/dynamic_topic_modeling_arxiv_abstracts](https://huggingface.co/datasets/ankitagr01/dynamic_topic_modeling_arxiv_abstracts) from Hugging Face is used. It consists of abstracts from arXiv and its corresponding topics. Selecting a dataset with topics for each text paragraph, rather than titles, summaries, or topic classes, is crucial for fine-tuning Llama 3. This ensures the model understands that it should dynamically generate only topics based on the content of each paragraph and not anything else.

### Data Preparation: Transforming Data Samples into Prompts
Llama 3 is pre-trained on vast amounts of text data to understand natural language. However, to optimize its performance for specific tasks like dynamic topic generation, it needs to be fine-tuned using datasets formatted as prompts. This process provides additional context and improve its ability to generate text relevant to the task. Below is an example of original sample transformed to the prompt format.
  ```bash
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Please generate a meaningful topic for the following article.
    
    ### Input:
    Maximum Variance Unfolding (MVU) and its variants have been very successful in embedding data-manifolds in lower dimensional spaces, often revealing the true intrinsic dimension. In this paper we show how to also incorporate       supervised class information into an MVU-like method without breaking its convexity. We call this method the Isometric Separation Map and we show that the resulting kernel matrix can be used as a binary/multiclass Support Vector Machine-like method in a semi-supervised (transductive) framework. We also show that the method always finds a kernel matrix that linearly separates the training data exactly without projecting them in infinite dimensional spaces. In traditional SVMs we choose a kernel and hope that the data become linearly separable in the kernel space. In this paper we show how the hyperplane can be chosen ad-hoc and the kernel is trained so that data are always linearly separable. Comparisons with Large Margin SVMs show comparable performance.
    
    
    ### Response:
    Semi-supervised Learning with MVU
"""
  ```

## Models
Considering the available resources, the pre-quantized 4-bit [unsloth/llama-3-8b-bnb-4bit](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) model, optimized for memory efficiency is used. The model is fine-tuned using [LoRA (Low-Rank Adaptation)](https://www.entrypointai.com/blog/lora-fine-tuning/), which updates only a small fraction of parameters. The [unsloth](https://www.unsloth.ai/blog/llama3) configuration provides improved performance with faster processing, support for longer context lengths, and reduced VRAM usage.
 
## Prerequisites
* Linux or macOS (recommended)
* Python 3
* NVIDIA GPU + CUDA CuDNN

## Code Structure
The project is organized into the following directories and files:
- `docs/`: Contains information documents related to the project.
- `data/`: Contains input.txt file with sample text for testing purpose.
- `src/`: Includes the main source code files.
    - `models/`: Directory for model-related functionalities.
        - `model_handler.py`: Script to load and save a model
        - `topic_generation_model.py`: Script to train dynamic topic generation model.
    - `scripts/`: Directory for additional scripts and utilities.
        - `data_format.py`: Script to handle data formatting.
    - `inference.py`: Script for model inference.
    - `train.py`: Script for model training.
- `dockerignore`: To exclude unnecessary files from the docker build context.
- `gitignore`: To ignore certain files from version control.
- `Dockerfile`: To build the project's docker image.
- `docker-compose.yml`: Configuration file for docker compose.
- `environment.yml`: Defines the conda environment for the project.
- `requirements.txt`: Lists python dependencies required for the project.
- `task.txt`: Task-related information.
  
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation
* To clone this repo:
  ```bash
  git clone https://github.com/meghanaraobn/dynamic-topic-generation-llama3.git
  cd dynamic-topic-generation-llama3
  ```
* For pip users, please type the command `pip install -r requirements.txt`.
* For conda users, you can create a new conda environment using `conda env create -f environment.yml`.
### Docker Support
This project includes docker support for easy setup with configurations provided in `docker-compose.yml` and `Dockerfile`. To enable GPU support within Docker containers, ensure the NVIDIA Container Toolkit is installed on the system. Detailed installation instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

For managing the setup:
* Build the docker image:
  ```bash
  docker-compose build
  ```
* To spin the container:
  ```bash
  docker-compose up -d
  ```
* To enter into the container (topic-generation-container):
  ```bash
  docker exec -it topic-generation-container bash
  ```
* To stop the container:
  ```bash
  docker-compose down
  ```

## Fine-tuning the model
 Logging is done using [Weights & Biases](https://wandb.ai/site). An account should be created to log the experiments.

 To fine-tune the pre-trained Llama 3 model, follow these steps:
 * View all available arguments and their default values:
   ```bash
   python src/train.py --help
   ```
  * Fine-tune with default settings:
    ```bash
    python src/train.py
    ```
  * Example of setting specific arguments:
    ```bash
    python src/train.py --wandb_key '<api_key>' --num_train_epochs 1 --model_save_path 'fine_tuned_model'
    ```
    Note: To use steps instead of epochs for fine-tuning, comment out `num_train_epochs` and uncomment the `max_steps` training argument in the code.

    Fine-tuned model can be downloaded [here](https://1drv.ms/u/s!ApQWtR4hn8DdsfZH6-p0EQ-0xkXsEQ?e=8XEfmb).

 ## Model Inference
 To dynamically generate topics for a given input text collection, follow these steps:
 * View all required arguments:
   ```bash
   python src/inference.py --help
   ```
 * Prepare your input:
   Place the input text in the file `data/input.txt` and save the file.
   
 * Generate topics from input:
   ```bash
   python src/inference.py --model_path 'fine_tuned_model' --input_file 'data/input.txt'
   ```
   Note: model_path should point to the directory containing the fine-tuned model.


## Notebook
The notebook version of the project is present [here](./src/notebooks/Dynamic_Topic_Generation_Llama_3.ipynb).

## Improvement Ideas and Challenges
The ideas for improving model performance, evaluation metrics and addressing challenges is present [here](./docs/improvement_ideas_and_evaluation_metrics_challenges.txt)
 
