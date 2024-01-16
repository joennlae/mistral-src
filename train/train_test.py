# source https://colab.research.google.com/drive/1DYY1zPxC-iotrfEu3TSkUBtfm1xJWR-i?usp=sharing#scrollTo=fNZZilUV_L4r

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, gradio, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login

base_model, dataset_name, new_model = (
    "mistralai/Mistral-7B-v0.1",
    "gathnex/Gath_baize",
    "gathnex/Gath_mistral_7b",
)

# Importing the dataset
dataset = load_dataset(
    dataset_name, split="train", cache_dir="/scratch/janniss/datasets"
)
print(dataset["chat_sample"][0])

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0},
    cache_dir="/scratch/janniss/huggingface",
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model, trust_remote_code=True, cache_dir="/scratch/janniss/huggingface"
)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)
model = get_peft_model(model, peft_config)

# Monitering the LLM
wandb.login(key="b65bd63fdcb4fa3b9b40c00b80096663d4c24045")
run = wandb.init(
    project="Fine tuning mistral 7B", job_type="training", anonymous="allow"
)

# Hyperparamter
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    save_steps=1000,
    logging_steps=30,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="chat_sample",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()


def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = "The conversation between Human and AI assisatance named Gathnex\n"
    B_INST, E_INST = "[INST]", "[/INST]"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)


stream("what is newtons 2rd law and its formula")
