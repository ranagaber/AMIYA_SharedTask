from configs import *
from functools import partial
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM , TrainerCallback , default_data_collator , Trainer, TrainingArguments , set_seed , logging , AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datasets import Dataset , concatenate_datasets
from peft import LoraConfig, get_peft_model , prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from huggingface_hub import login , upload_folder
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

EN_DA = ""
MSA_DA = ""
QA =  ""
repo_id = ''
full_model_path = './full_model'
model_token = ''  


login(token = "token if needed")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

country_map = {
    "Morocco" : "Moroccan Arabic",
    "Egypt" : "Egyptian Arabic",
    "Palestine" : "Palestinian Arabic",
    "Saudi Arabia" : "Saudi Arabic",
    "Syria" : "Syrian Arabic"
}

def load_MT_data(directory, source_col, source_language):
    datasets = []
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")])
    #print("Files found:", files)

    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns={source_col: 'source_text'})
        df = df.reset_index(drop=True)
        df['target'] = None
        df['source_language'] = source_language

        def make_input(row):
            target_language = country_map.get(row['Country'], row['Country'])
            if row.name % 2 == 0:
                prompt = f"Translate from {row['source_language']} into {target_language}, Output only the translation, do NOT output anything else before nor after it."
                return pd.Series([prompt + str(row["source_text"]), row["Text_DA"]])
            else:
                prompt = f"translate from {target_language} into {row['source_language']}, Output only the translation, do NOT output anything else before nor after it."
                return pd.Series([prompt + str(row["Text_DA"]), row["source_text"]])

        df[['input', 'target']] = df.apply(make_input, axis=1)
        datasets.append(Dataset.from_pandas(df))

    return concatenate_datasets(datasets)

def load_QA_data(directory):
    datasets  = []
    cols_to_drop = ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0','Unnamed: 0.3' , 'dialect']
    csv_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")])
    for f in csv_files:
        df = pd.read_csv(f , encoding = "utf-8-sig")
        if df.columns.intersection(cols_to_drop).tolist():
            df = df.drop(columns=df.columns.intersection(cols_to_drop).tolist())
        datasets.append(Dataset.from_pandas(df))
    return concatenate_datasets(datasets)

eng_dataset = load_MT_data(EN_DA, "Text_EN", "English")
msa_dataset = load_MT_data(MSA_DA ,"Text_MSA" ,"Standard Arabic")
QA_dataset = load_QA_data(QA) 

eng_dataset = eng_dataset.remove_columns(['Text_DA', 'Country', 'Dataset', 'SentID', 'source_text', 'source_language',])
msa_dataset = msa_dataset.remove_columns(['Text_DA', 'Country', 'Dataset', 'SentID', 'source_text', 'source_language',])

dataset = concatenate_datasets([eng_dataset, msa_dataset , QA_dataset]).shuffle(seed=SEED)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True )
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)



if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def preprocess(examples, max_length=256):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for prompt, target in zip(examples["input"], examples["target"]):
        prompt = str(prompt) if prompt is not None else ""
        target = str(target) if target is not None else ""
        messages = [{"role": "user", "content": prompt}] 
        prompt = tokenizer.apply_chat_template( 
            messages,
            tokenize=False,
            add_generation_prompt=True   
        )

        target_text = target + tokenizer.eos_token

        max_prompt_len = max_length - 32
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_len, return_tensors="pt")["input_ids"][0]

        max_target_length = max_length - len(prompt_ids)
        target_ids = tokenizer(target_text, add_special_tokens=False, truncation=True, max_length=max_target_length, return_tensors="pt")["input_ids"][0]

        # Truncate
        min_target_len = 1
        if len(prompt_ids) + len(target_ids) > max_length:
            min_target_len = 1
            max_prompt_len = max(max_length - min_target_len, 1)
            prompt_ids = prompt_ids[:max_prompt_len]

            available_target_len = max(max_length - len(prompt_ids), 1)
            target_ids = target_ids[:available_target_len]

        prompt_ids_list = prompt_ids.tolist()
        target_ids_list = target_ids.tolist()

        input_ids = prompt_ids_list + target_ids_list
        labels = [-100] * len(prompt_ids_list) + target_ids_list

        # Attention mask
        attention_mask = [1] * len(input_ids)

        # Pad
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }

tokenized_train = dataset.map(preprocess , batched = True )

data_collator = default_data_collator

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    save_strategy="steps",
    save_steps= SAVE_STEPS,
    seed = SEED,
    learning_rate= lr,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=VAL_BATCH,
    gradient_accumulation_steps= GRAD_ACCUM,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    fp16=False,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    report_to="wandb",
)



#model.config.use_cache = False
#model.gradient_checkpointing_enable()
#model.enable_input_require_grads()

class PercentCheckpointCallback(TrainerCallback):
    def __init__(self, tokenizer, checkpoints=[0.3, 0.5, 0.7]):
        super().__init__()
        self.checkpoints = checkpoints
        self.saved = set()
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, **kwargs):
        current_progress = state.global_step / max(1, state.max_steps)
        for cp in self.checkpoints:
            if current_progress >= cp:
                self.saved.add(cp)

    def on_step_end(self, args, state, control, **kwargs):
        progress = state.global_step / max(1, state.max_steps)
        for cp in self.checkpoints:
            if progress >= cp and cp not in self.saved:
                folder_name = os.path.join(args.output_dir, f"checkpoint_{int(cp*100)}")
                
                if not os.path.exists(folder_name):
                    print(f"Saving checkpoint at {int(cp*100)}% -> {folder_name}")
                    kwargs['model'].save_pretrained(folder_name)
                    self.tokenizer.save_pretrained(folder_name)
                    self.saved.add(cp)
        return control


callback = [PercentCheckpointCallback(tokenizer=tokenizer, checkpoints=[0.3,0.5,0.7])]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callback,
)
trainer.train()

'''
checkpoint = ''
torch.load = partial(torch.load, weights_only=False)
trainer.train(resume_from_checkpoint = checkpoint)
'''

trainer.save_model(full_model_path)


upload_folder(folder_path=full_model_path, repo_id= repo_id , repo_type="model",  token = model_token)