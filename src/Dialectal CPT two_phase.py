from configs import MODEL_CONFIGS, BATCH_SIZE, GRAD_ACCUM
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator, TrainerCallback, set_seed
from datasets import load_dataset, concatenate_datasets
import torch
import os
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PHASE1_EPOCHS = 1
PHASE2_EPOCHS = 1

EN_DIR = "EN"
MSA_DIR = "MSA"
DA_DIR = "DA"

class Phase2CheckpointCallback(TrainerCallback):
    """Save a checkpoint at a fraction of Phase 2 training."""
    def __init__(self, save_fraction=0.3, output_dir=None):
        self.save_fraction = save_fraction
        self.saved = False
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if self.saved:
            return
        total_steps = state.max_steps
        target_step = int(total_steps * self.save_fraction)
        if state.global_step >= target_step:
            control.should_save = True
            print(f"Saving Phase 2 checkpoint at step {state.global_step} ({self.save_fraction*100}%)")
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                kwargs['model'].save_pretrained(self.output_dir)
            self.saved = True


def load_all_csvs_from_dir(directory):
    csv_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")])
    datasets = [load_dataset("csv", data_files=f)['train'] for f in csv_files]
    return concatenate_datasets(datasets)


def filter_data(example):
    return example.get("Dataset") != "madar" and example.get("Text") not in (None, "")


def run_phase(model, train_dataset, output_dir, peak_lr, warmup_ratio, epochs,
              tokenizer=None, resume_from_checkpoint=None, callbacks=None, seq_len=256):
    
    os.makedirs(output_dir, exist_ok=True)

    def tokenize(batch):
        texts = [(text or "") + tokenizer.eos_token for text in batch["Text"]]
        tokens = tokenizer(
            texts,               
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        input_ids = torch.tensor(tokens["input_ids"])
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokens["input_ids"] = input_ids
        tokens["labels"] = labels
        return tokens

    train_dataset = train_dataset.map(lambda x: tokenize(x), batched=True, batch_size=5000, num_proc=4, remove_columns=["Text"])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=peak_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_total_limit=1,
        save_steps=500,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model = trainer.model
    return model


def run_two_phase_cpt(model_id, dataset_phase1, dataset_phase2):
    cfg = MODEL_CONFIGS[model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

     model = AutoModelForCausalLM.from_pretrained(
         model_id,
         device_map="auto",
         dtype=torch.bfloat16,
         trust_remote_code=True,
         use_safetensors=True,
         low_cpu_mem_usage=True,
     )
     model.config.use_cache = False

     phase1_dir = f"ckpts/{model_id.replace('/', '_')}_phase1"
     phase1_ckpt = None
     if os.path.exists(phase1_dir):
         subdirs = [os.path.join(phase1_dir, d) for d in os.listdir(phase1_dir)
                    if os.path.isdir(os.path.join(phase1_dir, d))]
         if subdirs:
             phase1_ckpt = sorted(subdirs)[-1]
     model = run_phase(
         model=model,
         train_dataset=dataset_phase1,
         output_dir=phase1_dir,
         peak_lr=cfg["phase1_lr"],
         warmup_ratio=cfg["warmup_ratio"],
         epochs=PHASE1_EPOCHS,
         tokenizer=tokenizer,
         resume_from_checkpoint=phase1_ckpt
     )
    model.save_pretrained(phase1_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        phase1_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    
    phase2_dir = f"ckpts/{model_id.replace('/', '_')}_phase2"
    phase2_ckpt = None
    if os.path.exists(phase2_dir):
        subdirs = [os.path.join(phase1_dir, d) for d in os.listdir(phase1_dir)
                   if os.path.isdir(os.path.join(phase1_dir, d))]
        if subdirs:
            phase1_ckpt = sorted(subdirs)[-1]

    phase2_callback = [Phase2CheckpointCallback(save_fraction=0.3,
                                                output_dir=os.path.join(phase2_dir, "30pct"))]
    model = run_phase(
        model=model,
        train_dataset=dataset_phase2,
        output_dir=phase2_dir,
        peak_lr=cfg["phase2_lr"],
        warmup_ratio=cfg["warmup_ratio"],
        epochs=PHASE2_EPOCHS,
        tokenizer=tokenizer,
        resume_from_checkpoint=phase2_ckpt,
        callbacks=phase2_callback
    )
    model.save_pretrained(phase2_dir)

    return model


if __name__ == "__main__":
    dataset_en = load_all_csvs_from_dir(EN_DIR).shuffle(seed=42).select(range(100_000))
    dataset_msa = load_all_csvs_from_dir(MSA_DIR)
    dataset_da = load_all_csvs_from_dir(DA_DIR)

    dataset_phase1 = concatenate_datasets([dataset_en, dataset_msa]).shuffle(seed=42)
    dataset_phase2 = dataset_da.shuffle(seed=42)

    dataset_phase1 = dataset_phase1.filter(filter_data)
    dataset_phase2 = dataset_phase2.filter(filter_data)

    print("Phase 1 dataset length after filtering:", len(dataset_phase1))
    print("Phase 2 dataset length after filtering:", len(dataset_phase2))

    for model_id in MODEL_CONFIGS:
        run_two_phase_cpt(
            model_id=model_id,
            dataset_phase1=dataset_phase1,
            dataset_phase2=dataset_phase2,
        )
