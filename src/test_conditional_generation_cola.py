# environments: pip install -q transformers dataset evaluate seqeval accelerate safetensors

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from module.peft import (
    ColaConfig,
    TaskType,
    Metric,
    get_peft_model, 
    save_intermediate_info, 
    create_gradient_boosting_models, 
    create_gradient_boosting_datasets,
    save_gradient_boosting_models,
    create_optimizer,
    create_scheduler,
    make_data_loader,
    collate,
    to_device,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# device = "cuda"
device = "cpu"
model_name_or_path = "facebook/bart-base"
tokenizer_name_or_path = "facebook/bart-base"

checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-3
num_epochs = 1
batch_size = 8

gradient_boosting_cfg = {
    'optimizer': 'SGD',
    'scheduler': 'CosineAnnealingLR',
    'batch_size': {'train': 2},
    'shuffle': {'train': True},
    'seed': 0,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'nesterov': True,
    'pin_memory': True,
    'num_workers': 0,
    'max_clip_norm': 10,
    'optimizer_name': 'SGD',
    'scheduler_name': 'CosineAnnealingLR',
    'num_epochs': 100,
}

peft_config = ColaConfig(
    base_model_name_or_path=model_name_or_path,
    task_type=TaskType.SEQ_2_SEQ_LM,
    dataset_name='financial_sentiment_analysis',
    r=64,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    get_delta_h=True
)

cache_model_path = os.path.join('output', 'model', 'bart-base')
cache_tokenizer_path = os.path.join('output', 'tokenizer', 'bart-base')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_model_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)


# data preprocessing
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
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

model.base_model.peft_config['default'].total_step = len(train_dataloader) * num_epochs

# training and evaluation
model = model.to(device)
global_step = 0
for epoch in range(1):
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
        optimizer.zero_grad()
        global_step += 1
        # test workflow faster
        if global_step == 2:
            break

    save_intermediate_info(
        peft_config=peft_config, 
        model=model, 
        save_mode='overwrite_mode'
    )

metric = Metric({'train': ['Loss', 'MAD'], 'test': ['Loss', 'MAD']})
for epoch in range(num_epochs):
    model.eval()
    eval_loss = 0
    eval_preds = []
    gradient_boosting_models = create_gradient_boosting_models(peft_config, model)
    gradient_boosting_datasets = create_gradient_boosting_datasets(peft_config)

    # train gradient boosting model
    for key, model in gradient_boosting_models.items():
        data_loader = make_data_loader(
            {'train': gradient_boosting_datasets[key]}, 
            gradient_boosting_cfg
        )
        optimizer = create_optimizer(model, gradient_boosting_cfg)
        scheduler = create_scheduler(optimizer, gradient_boosting_cfg)
        for i, input in enumerate(data_loader['train']):
            input = collate(input)
            # print(f"input[id]: {input['id']}\n")
            input_size = input['data'].size(0)
            input = to_device(input, device)
            optimizer.zero_grad()
            output = model(input)
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_boosting_cfg['max_clip_norm'])
            optimizer.step()
            evaluation = metric.evaluate(
                metric.metric_name['train'], 
                input, 
                output
            )

    # save gradient boosting model
    save_gradient_boosting_models( 
        peft_config=peft_config,
        models=gradient_boosting_models,
    )

    # load gradient boosting model to LLM when inference
    peft_config = ColaConfig(
        base_model_name_or_path=model_name_or_path,
        task_type=TaskType.SEQ_2_SEQ_LM,
        dataset_name='financial_sentiment_analysis',
        r=64,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
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
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{dataset['validation']['text_label'][:10]=}")



model.eval()
i = 13
inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))