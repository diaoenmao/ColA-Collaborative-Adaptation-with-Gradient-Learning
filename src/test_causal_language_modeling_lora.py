# environments: pip install -q transformers dataset evaluate seqeval accelerate safetensors

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from module.peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda"
# device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"

# checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
# text_column = "sentence"
# label_column = "text_label"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64

lr = 3e-2
num_epochs = 50
batch_size = 8

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)


# a = model.base_model
# loading dataset
dataset = load_dataset("ought/raft", "twitter_complaints")
classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
print(dataset["train"].column_names)
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

cache_model_path = os.path.join('output', 'model', 'bloomz')
cache_tokenizer_path = os.path.join('output', 'tokenizer', 'bloomz')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# classes = dataset["train"].features["label"].names


# data preprocessing
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# tokenizer.pad_token_id = 3
def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# b = model.base_model
# model.base_model.peft_config['default'].total_step = len(train_dataloader) * num_epochs


# training and evaluation
model = model.to(device)
global_step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # Update the importance of low-rank matrices
        # and allocate the budget accordingly.
        # model.base_model.update_and_allocate(global_step)
        optimizer.zero_grad()
        global_step += 1

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(train_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(eval_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


# print accuracy
# correct = 0
# total = 0
# for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
#     if pred.strip() == true.strip():
#         correct += 1
#     total += 1
# accuracy = correct / total * 100
# print(f"{accuracy=} % on the evaluation dataset")
# print(f"{eval_preds[:10]=}")
# print(f"{dataset['validation']['text_label'][:10]=}")





# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)


ckpt = f"{peft_model_id}/adapter_model.bin"
# get_ipython().system('du -h $ckpt')


peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)


model.eval()
i = 4
# inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
inputs = tokenizer(f'{text_column} : {dataset["validation"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    # outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                             max_new_tokens=10, eos_token_id=3)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))