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

