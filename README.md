# Dynamic Topic Generation using Llama 3
Dynamic topic generation is the process of automatically identifying and categorizing the main themes or subjects within a collection of text data. Unlike traditional methods that rely on predefined categories, dynamic topic generation adapts to the content being analyzed, allowing topics to vary based on the unique characteristics of the texts.

### Why Llama 3? 
<img src="https://github.com/meghanaraobn/SUMM-AI-TASK/assets/76180138/675b0481-ae76-48bf-8936-6e99c253356c" width="20%" height="15%" align="right" />

[Llama 3](https://ai.meta.com/blog/meta-llama-3/), a Large Language Model (LLM), is chosen for this task due to its robust capabilities in understanding and generating natural language text. Llama 3 can effectively capture the semantic meaning and context from text paragraphs, making it suitable for dynamic topic generation. a Large Language Model (LLM), is chosen for this task due to its robust capabilities in understanding and generating natural language text. Llama 3 can effectively capture the semantic meaning and context from text paragraphs, making it suitable for dynamic topic generation.

### Objective
The objective of this project is to fine-tune Llama 3 to perform dynamic topic generation for a given text paragraph.

## Datasets
The dataset [ankitagr01/dynamic_topic_modeling_arxiv_abstracts](https://huggingface.co/datasets/ankitagr01/dynamic_topic_modeling_arxiv_abstracts) from Hugging Face is used. It consists of abstracts from arXiv and its corresponding topics. Selecting a dataset with topics for each text paragraph, rather than titles, summaries, or topic classes, is crucial for Llama 3. This ensures the model understands that it should dynamically generate topics based on the content of each paragraph.

## Models
The pre-quantized 4-bit [unsloth/llama-3-8b-bnb-4bit](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) model, optimized for memory efficiency is used. The model is fine-tuned using [LoRA (Low-Rank Adaptation)](https://www.entrypointai.com/blog/lora-fine-tuning/), which updates only a small fraction of parameters. The [unsloth](https://www.unsloth.ai/blog/llama3) configuration enhances performance with faster processing, support for longer context lengths, and reduced VRAM usage.

## Process
- Coming soon
## Prerequisites
* Linux or macOS (recommended)
* Python 3
* NVIDIA GPU + CUDA CuDNN
  
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation
* Clone this repo:
  ```bash
  git clone https://github.com/meghanaraobn/dynamic-topic-generation-llama3.git
  cd dynamic-topic-generation-llama3
  ```
* For pip users, please type the command `pip install -r requirements.txt`.
* For conda users, you can create a new conda environment using `conda env create -f environment.yml`.
### Docker Support
This project includes docker support for easy setup with configurations provided in `docker-compose.yml` and `Dockerfile`. To enable GPU support within Docker containers, ensure the NVIDIA Container Toolkit is installed on the system. Detailed installation instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

For managing the setup:
* Build the docker image
  ```bash
  docker-compose build
  ```
* To spin the container
  ```bash
  docker-compose up -d
  ```
* To enter into the container (topic-generation-container)
  ```bash
  docker exec -it topic-generation-container bash
  ```
* To stop the container
  ```bash
  docker-compose down
  ```

### Fine-tuning the model
 Logging is done using [Weights & Biases](https://wandb.ai/site). An account should be created to log the experiments.

 To fine-tune the pre-trained Llama 3 model, use the following commands:
 * View all available arguments and their default values:
   ```bash
   python train.py --help
   ```
  * Fine-tune with default settings:
    ```bash
    python train.py
    ```
  * Example of setting specific arguments:
    ```bash
    python train.py --wandb_key '<api_key>' --num_train_epochs 1 
    ```
    Note: Currently, number of steps for training is not enabled. To use steps instead of epochs, comment out `num_train_epochs` and uncomment the `max_steps` training argument in the code.

## Notebook
- Coming soon

## Sample results  
- Coming soon

## Improvement Ideas
- Coming soon
 
