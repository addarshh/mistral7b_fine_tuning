# utils/model_utils.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import torch

def initialize_model(model_name, device_map='auto', quantization_config=None, use_cache=False):
    """
    Initialize the model and tokenizer for training or inference.

    Args:
    - model_name: The name or path of the pre-trained model to load.
    - device_map: The device placement strategy for the model. 'auto' will automatically place the layers.
    - quantization_config: Configuration for model quantization.
    - use_cache: Whether to use caching of model outputs.

    Returns:
    - model: The loaded and optionally quantized model.
    - tokenizer: The tokenizer associated with the model.
    """
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    nf8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )
    quantization_config = nf8_config
    # Load model with potential quantization configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quantization_config,
        use_cache=use_cache
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, tokenizer

def save_model(model, path):
    """
    Save a trained model to the specified path.

    Args:
    - model: The model to save.
    - path: The path where the model should be saved.
    """
    model.save_pretrained(path)

def load_model(path):
    """
    Load a model from the specified path.

    Args:
    - path: The path from where to load the model.

    Returns:
    - The loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(path)
    return model


def train_model(model, train_dataset, eval_dataset, tokenizer, output_dir="model_output", training_args=None):
    """
    Train the model using the provided datasets.

    Args:
    - model: The model to be fine-tuned.
    - train_dataset: The training dataset.
    - eval_dataset: The evaluation dataset.
    - tokenizer: The tokenizer to use for encoding the datasets.
    - output_dir: Directory to save the model after training.
    - training_args: An instance of transformers.TrainingArguments to customize training behavior.

    Returns:
    - The trained model.
    """

    if training_args is None:
        # Define default training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory for saving model checkpoints
            num_train_epochs=3,             # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=8,   # batch size for evaluation
            warmup_steps=500,               # number of warmup steps for learning rate scheduler
            weight_decay=0.01,              # strength of weight decay
            logging_dir='./logs',           # directory for storing logs
            logging_steps=10,               # log & save weights each logging_steps
            evaluation_strategy="steps",    # evaluation is done (and logged) every eval_steps
            eval_steps=100,                 # evaluate every 100 steps
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,    # load the best model when finished training (default metric is loss)
            report_to="wandb",
            max_seq_length=2048
            # Add more arguments as needed
        )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Define compute_metrics function for evaluation if needed
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer, trainer


def push_model_to_hub(model, tokenizer, trainer, repo_name):
    """
    Pushes the trained model and tokenizer to the Hugging Face Hub.

    Args:
    - model: The trained model object.
    - tokenizer: The associated tokenizer.
    - model_name: The name to give to the model on the hub.
    - organization: Optional. The organization under which to push the model. If None, pushes under the user's namespace.

    Returns:
    - None
    """
    model = model.merge_and_unload()
    
    # Push model to the hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    trainer.push_to_hub(repo_name)
    
    print(f"Model and tokenizer pushed to the hub under {repo_name}")