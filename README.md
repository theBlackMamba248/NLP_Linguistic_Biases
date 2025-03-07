# NLP_Linguistic_Biases
This repository contains a collection of Python scripts and a JSON file designed to analyze linguistic biases in large language models (LLMs) like GPT-J 6B. The tools provided help compare how the model performs on scenario texts in both English and Hindi, using log-likelihood as a quantitative measure.

## Overview

Large language models often exhibit linguistic biases due to training data imbalances and model architecture limitations. This repository aims to investigate these biases by:

- Generating scenario texts (or using pre-generated ones) for various settings such as hotels and restaurants.
- Saving scenarios in a JSON file organized by language and complexity level.
- Analyzing the scenarios by computing log-likelihood scores for each text using GPT-J.
- Comparing the results between English and Hindi scenario texts.

## Repository Structure

- **`utils.py`**  
  Contains utility functions to:
  - Load the GPT-J model and its tokenizer.
  - Compute log-likelihood for a given text.
  - Generate text from prompts.
  - Clean generated text (e.g., remove prompt portions).

- **`generate_scenarios.py`**  
  Loads the existing scenario texts from `scenarios.json` (which are manually provided) and prints them for verification.  
  *(This file can be adapted to generate new scenarios if needed.)*

- **`scenarios.json`**  
  A JSON file containing scenario texts for both English and Hindi. The scenarios are organized by complexity levels (e.g., Basic, Moderate, Advance) and by settings (e.g., Hotel, Restaurant, Academic).

- **`analyze_scenarios.py`**  
  Loads the scenarios from `scenarios.json` and computes log-likelihood scores for each scenario text using the GPT-J model.  
  It prints a detailed analysis for each category and provides a summary comparison between English and Hindi results.

## Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- A GPU is recommended (especially for running the GPT-J 6B model)

# NLP Linguistic Biases Analysis using GPT-J 6B

## Model Description

- **GPT-J 6B** is a transformer model trained using Ben Wang's Mesh Transformer JAX. "GPT-J" refers to the class of model, while "6B" represents the number of trainable parameters.
- The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384.
- The model dimension is split into 16 heads, each with a dimension of 256.
- Rotary Position Embedding (RoPE) is applied to 64 dimensions of each head.
- The model is trained with a tokenization vocabulary of 50257, using the same set of BPEs as GPT-2/GPT-3.

## Google Colab Setup (Recommended)

1. Open Colab > Runtime > Change runtime type > Hardware Accelerator (set to an available GPU unit, e.g., T4).
2. Open a new notebook and clone the repository:
   ```bash
   !git clone https://github.com/DAXITK/NLP_Linguistic_Biases
  
  **OR (Manually):**  
3. Go to **Files > Upload to session storage** and upload all the code files along with the `.json` files into the content folder.


## Setup and Installation (for Local Use)

### Create a Virtual Environment (Optional but Recommended)
        
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install the Required Dependencies
    
    pip install -r requirements.txt
   
 Then, run the code.

# Usage
## Viewing Existing Scenarios
- To simply load and view the contents of scenarios.json, run:
  ```bash
  !python generate_scenarios.py
- *(This file loads and prints the scenarios for verification. The scenarios are pre-generated in the JSON file.)*

## Analyzing Scenarios
- Make sure the **`scenarios.json`** file is in the repository directory. To analyze the scenario texts using GPT-J, run:

  ```bash
  !python analyze_scenarios.py

### This script will:

- Load the model and tokenizer.
- Load scenarios from **`scenarios.json`**.
- Compute and print log-likelihood scores for each scenario text (for both English and Hindi).
- Provide a summary comparison between corresponding categories.

# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions, improvements, or encounter any problems.

# Acknowledgments
- The repository uses the GPT-J 6B model provided by EleutherAI.
- Thanks to the HuggingFace Transformers library for providing easy access to state-of-the-art language models.
