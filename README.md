# Samantha - Emotional Companinion (Fine Tuning of Mistral 7B Model)

## Overview

This project involves fine-tuning a Fine Tuning Mistral-7B for digital companionship, Samantha, a character from the movie Her. Here, we first fine tune the mistral 7b model and then used the fine-tuned model to generate responses or perform another task as defined by the user. The project is structured into scripts for fine-tuning, using the model, and utility functions for dataset and model management.

## Data Used
This project utilizes the "samantha-data" dataset from Hugging Face, focused on conversational AI in areas like philosophy and psychology. The dataset is structured with training, validation, and testing splits, facilitating a robust training regimen. Explore the dataset [here](https://huggingface.co/datasets/cognitivecomputations/samantha-data).



## Project Structure
Below is the structure of the project directory:

- `fine_tune_model.py`: Script for fine-tuning the pre-trained model on a custom dataset.
- `use_model.py`: Script for using the fine-tuned model to generate responses or perform the task it was fine-tuned for.
- `requirements.txt`: Lists all the libraries and their versions required to run the project.
- `api_key.py`: Stores API keys or other sensitive information required for accessing certain resources or services.
- `utils/`: Directory containing utility modules.
  - `dataset_utils.py`: Provides utilities for handling the dataset, including loading, preprocessing, and splitting.
  - `model_utils.py`: Contains functions for model management, such as loading, saving, and configuring the model.

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.10 or later installed.
3. Install the required dependencies by running `pip install -r requirements.txt` in your project's root directory.

## Usage

### Fine-Tuning the Model

To fine-tune the model, run:

```bash
python fine_tune_model.py
```

### Fine-Tuning the Model

After fine-tuning, you can use the model to generate responses or perform the task it was trained for:

```bash
python use_model.py "<your_prompt_here>"
```
## Configuration

- Modify `api_key.py` to include your API keys or other sensitive data required by the project.
- You can adjust fine-tuning parameters and model configurations in `fine_tune_model.py`.
- For using the model, if additional configuration is required, modify `use_model.py` accordingly.

## Requirements

Please refer to `requirements.txt` for a detailed list of all necessary libraries. This project may require:

- `transformers`
- `torch`
- `numpy`
- (Any other libraries listed in your `requirements.txt`)

