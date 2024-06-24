# Dynamic Topic Generation using Llama 3
Dynamic topic generation is the process of automatically identifying and categorizing the main themes or subjects within a collection of text data. Unlike traditional methods that rely on predefined categories, dynamic topic generation adapts to the content being analyzed, allowing topics to vary based on the unique characteristics of the texts.

### Why Llama 3? 
<img src="https://github.com/meghanaraobn/SUMM-AI-TASK/assets/76180138/675b0481-ae76-48bf-8936-6e99c253356c" width="20%" height="15%" align="right" />
[Llama 3](https://ai.meta.com/blog/meta-llama-3/), a Large Language Model (LLM), is chosen for this task due to its robust capabilities in understanding and generating natural language text. Llama 3 can effectively capture the semantic meaning and context from text paragraphs, making it suitable for dynamic topic generation.

### Objective
The objective of this project is to fine-tune Llama 3 to perform dynamic topic generation for a given text paragraph.

## Datasets
The dataset [ankitagr01/dynamic_topic_modeling_arxiv_abstracts](https://huggingface.co/datasets/ankitagr01/dynamic_topic_modeling_arxiv_abstracts) from Hugging Face is used. It consists of abstracts from arXiv and its corresponding topics. Selecting a dataset with topics for each text paragraph, rather than titles, summaries, or topic classes, is crucial for Llama 3. This ensures the model understands that it should dynamically generate topics based on the content of each paragraph.

## Models
The pre-quantized 4-bit [unsloth/llama-3-8b-bnb-4bit](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) model, optimized for memory efficiency is used. The model is fine-tuned using [LoRA (Low-Rank Adaptation)](https://www.entrypointai.com/blog/lora-fine-tuning/), which updates only a small fraction of parameters. The [unsloth](https://www.unsloth.ai/blog/llama3) configuration enhances performance with faster processing, support for longer context lengths, and reduced VRAM usage.

## Process
- Coming soon

## Getting Started

### Docker
- Coming soon.    

### Fine-tuning
 - Coming soon

## Notebook
- Coming soon

## Sample results  
- Coming soon

## Improvement Ideas
- Coming soon
 
