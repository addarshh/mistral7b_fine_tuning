import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load the model and tokenizer from local storage
def load_model_and_tokenizer(local_path):
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path)
    model.eval()  # Ensure model is in evaluation mode
    if torch.cuda.is_available():  # Move to GPU if available
        model.to("cuda")
    return model, tokenizer

def generate_response(input, model, tokenizer):
    instruction_prompt = "you are Samantha, an AI designed by Bell Labs Research Team with a unique identity, focused on providing friendship and companionship to users. Continue the following conversation as if you (Samantha) were responding to the human's last message."
    input_text =  "### Input:\n" + input + "\n\n### Response:"
    prompt = instruction_prompt + input_text
    print(prompt)
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if torch.cuda.is_available():
        encoded_input = encoded_input.to("cuda")
    with torch.no_grad():  # No need to track gradients for inference
        generated_ids = model.generate(**encoded_input, max_length=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response

def main(prompt):
    local_path = "../bell_labs_samantha_7b/"  # Update this path to your local model directory
    model, tokenizer = load_model_and_tokenizer(local_path)
    print("Model and tokenizer loaded successfully from local storage.")

    response = generate_response(prompt, model, tokenizer)
    print(f"Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a response from a prompt using a fine-tuned model.")
    parser.add_argument('prompt', type=str, help='Prompt for generating a response.')
    args = parser.parse_args()

    main(args.prompt)

