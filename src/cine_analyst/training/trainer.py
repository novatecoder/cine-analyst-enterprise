import os
import sys
import types
import torch
from loguru import logger
from cine_analyst.common.config import settings

# --- Unsloth/Torch 호환성 패치 ---
if not hasattr(torch, "int1"):
    torch.int1 = torch.bool 

if not hasattr(torch, "_inductor"):
    torch._inductor = types.ModuleType("_inductor")
if not hasattr(torch._inductor, "config"):
    torch._inductor.config = types.ModuleType("config")
sys.modules["torch._inductor.config"] = torch._inductor.config
# --- 패치 끝 ---

def train_model(
    base_model: str = settings.BASE_MODEL_NAME,
    data_path: str = settings.PROCESSED_DATA_PATH,
    output_dir: str = settings.MODEL_SAVE_DIR,
    max_steps: int = 60
):
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
        # import wandb  # 직접 호출하지 않으므로 필요 시에만 사용
    except ImportError:
        logger.error("❌ 'unsloth' or 'trl' not found. Run 'poetry install --with train'")
        return

    # [수정] 수동 wandb.init() 대신 환경 변수로 프로젝트 설정
    os.environ["WANDB_PROJECT"] = "cine-analyst-enterprise"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint" # 필요 시 모델 체크포인트 업로드 활성화

    logger.info(f"🚀 Loading Base Model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=settings.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=settings.LOAD_IN_4BIT,
    )

    # LoRA 설정 적용
    model = FastLanguageModel.get_peft_model(
        model,
        r=settings.LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=settings.LORA_ALPHA,
        lora_dropout=settings.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )

    # 데이터셋 로드 및 분할
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    logger.info(f"🔥 Starting SFT Training (Eval included)...")
    
    # Trainer 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
            report_to="wandb",          # [중요] Trainer가 직접 wandb 세션을 관리함
            run_name=f"train-{settings.ENV}-{max_steps}steps",
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    # 학습 시작
    trainer.train()
    
    # [수정] 학습 종료 후 별도의 evaluate() 호출은 에러가 발생할 수 있으므로 제거하거나, 
    # 필요한 경우 Trainer의 세션이 유지되는 동안 호출되어야 합니다.
    # 이미 load_best_model_at_end로 최적 모델이 로드된 상태입니다.

    logger.info(f"💾 Saving best adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # [수정] wandb.finish() 삭제 (Trainer가 자동으로 종료 처리)

def run_cli():
    train_model()