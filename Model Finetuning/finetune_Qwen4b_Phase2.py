import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from peft import PeftModel

# Login to Hugging Face
login("")

print("Program started")
print("CUDA available:", torch.cuda.is_available())
print("Total GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

torch.cuda.set_device(0)  # Set to 1 if you want the second GPU
print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Model parameters
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model configuration
model_name = "Qwen/Qwen3-4B"
adapter_repo = "deshanksuman/Qwen3-4B-MWSD-3epc"
max_seq_length = 1024
load_in_4bit = False
access_token = ""

# Configure quantization
if load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
else:
    quantization_config = None

# Load tokenizer directly from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=access_token,
    padding_side="right",
    use_fast=True,
)

# For Qwen models, pad_token should be set to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model directly from Hugging Face with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=False
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_repo, token=access_token)
print("Model loaded successfully!")

# ChatML format for Qwen2.5-3B-Instruct
def formatting_prompts_func(examples):
    words = examples["word"]
    sentences = examples["sentence"]
    poses = examples["pos"]
    meanings = examples["meaning"]
    lans = examples["language"]
    meaning_infos = examples["meaning_info"]
    senseIDs = examples["senseid"]
    
    texts = [] 

    # Additional context sentence to prepend to each input
    for word, sentence, pos, meaning, lan, meaning_info, senseID in zip(words, sentences, poses, meanings, lans, meaning_infos, senseIDs):

        additional_context = (
            f"You are an expert in understanding the word meaning of sentences given in {lan}. "
            f"The sentence contains an ambiguous word enclosed with <WSD> tags. "
            f"Identify the meaning of the word based on the context of the sentence."
            f"Return the correct sense ID related to the meaning."
        )
        
        # Create conversation using ChatML format
        conversation = [
            {
                "role": "system",
                "content": f"You are a helpful assistant specialized in word sense disambiguation for {lan}."
            },
            {
                "role": "user", 
                "content": (
                    f"{additional_context}\n"
                    f"The sentence is: {sentence}. "
                    f"The word to disambiguate is {word}. "
                    f"The possible sense ID and meaning options are: {meaning_info}. "
                    f"First identify the Part of Speech (PoS) tag and consider the surrounding words to identify the most suitable meaning. "
                    f"Return only the correct sense ID related to the meaning as the final answer."
               )
            },
            {
                "role": "assistant",
                "content": (
                    f"Based on the given sentence, the ambiguous word is {word}. "
                    f"The PoS of {word} is {pos}. "
                    f"Now let's analyze the meaning. Based on the surrounding words, the meaning is {meaning}."
                    f"So, the most appropriate sense ID is {senseID}."
                )
            }
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

# Load and prepare dataset
dataset = load_dataset("deshanksuman/senseMapped_MWSD_Train", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=2,
    learning_rate=1e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs_phase2",
    report_to="none",
    save_steps=1000,
    save_total_limit=2,
)


# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,              
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)


# Train the model
print("Starting training...")
trainer_stats = trainer.train()
print("Training completed!")


# Save the model
print("\nSaving model...")
model_name_hub = "deshanksuman/Qwen3-4B-MWSD-phase2"

# Save locally first (good practice)
model.save_pretrained("./lora_adapters")
tokenizer.save_pretrained("./lora_adapters")

# Push to Hugging Face Hub
model.push_to_hub(model_name_hub, token=access_token)
tokenizer.push_to_hub(model_name_hub, token=access_token)

print(f"Model successfully saved and pushed to: {model_name_hub}")
print("Training and saving completed successfully!")