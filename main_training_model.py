import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
)
from datasets import Dataset
from data_processing_glaff import DataProcessingGlaff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help = "Path to GLAFF txt files")
    parser.add_argument("--output_dir", type=str, default= "./byt5-g2p-French")
    parser.add_argument("--model_save_path", type=str, default="./french_g2p_model_final")
    processor = DataProcessingGlaff()
    df = processor.read_csv(args.data_path)
    merged_data = processor.build_pairs(df)

    device = "cuda" if torch.backends.mps.is_available() else "cpu"

    model_name = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    raw_dataset = Dataset.from_list(merged_data)

    def preprocess_function(examples):
        return tokenizer(
            examples["grapheme"],
            text_target=examples["phoneme"],
            max_length=64,
            truncation=True,
        )

    tokenized_ds = raw_dataset.map(
        preprocess_function, batched=True,
        remove_columns=raw_dataset.column_names,
    )

    split = tokenized_ds.train_test_split(test_size=0.1)
    train_dataset, eval_dataset = split["train"], split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/content/drive/MyDrive/byt5-g2p-french",
        eval_strategy="epoch",
        save_strategy ="epoch",
        load_best_model_at_end = True,
        metric_for_best_model ="eval_loss",
        learning_rate=4e-4,
        warmup_ratio=0.05,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=False,
        logging_steps=100,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  
        data_collator=data_collator, 
    )

    trainer.train()
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    main()

