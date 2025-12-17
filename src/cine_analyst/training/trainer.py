import os
from loguru import logger
from cine_analyst.common.config import settings

def train_model(
    base_model: str = settings.BASE_MODEL_NAME,
    data_path: str = settings.PROCESSED_DATA_PATH,
    output_dir: str = settings.MODEL_SAVE_DIR,
    max_steps: int = 60
):
    """
    Unsloth를 이용한 LoRA Fine-tuning.
    Service 의존성과 분리하기 위해 함수 내부에서 import 수행.
    """
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError:
        logger.error("❌ 'unsloth' or 'trl' not found. Run 'poetry install --with train'")
        return

    logger.info(f"🚀 Loading Base Model: {base_model}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=settings.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=settings.LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=settings.LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Chat Template Formatting
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    logger.info("🔥 Starting SFT Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=settings.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer.train()
    
    logger.info(f"💾 Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def run_cli():
    train_model()