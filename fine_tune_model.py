# fine_tune_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataset_utils import load_and_prepare_dataset
from utils.model_utils import initialize_model, train_model, push_model_to_hub
from api_key import WANDB_API_KEY, HF_USER_TOKEN

# Set up WANDB
import wandb
wandb.login(key=WANDB_API_KEY)

def fine_tune():
    # Load and prepare dataset
    dataset = load_and_prepare_dataset("wangqi777/samantha-data")
    print("Dataset loaded and prepared.")

    # Initialize model and tokenizer
    model, tokenizer = initialize_model("mistralai/Mistral-7B-v0.1")
    print("Model and tokenizer initialized.")

    # Train model
    model, tokenizer, trainer = train_model(model, dataset, tokenizer)
    print("Model training complete.")

    repo_name = "adarsh12x/mistral_7b_samantha_v1.2"
    push_model_to_hub(model, tokenizer, trainer, repo_name)

    

if __name__ == "__main__":
    fine_tune()
