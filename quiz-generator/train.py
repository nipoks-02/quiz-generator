from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset


model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")


dataset = load_dataset("json", data_files="data/quiz_dataset.json", split="train")


def preprocess(examples):
    inputs = [f"generate quiz: {topic}" for topic in examples["topic"]]
    outputs = [f"question: {q} answer: {a} explanation: {e}"
               for q, a, e in zip(examples["question"], examples["answer"], examples["explanation"])]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess, batched=True)


training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="logs",
    logging_steps=10,
    report_to="tensorboard"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("models/final_model")