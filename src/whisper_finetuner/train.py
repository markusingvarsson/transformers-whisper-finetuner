from datasets import Features, Value, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import librosa
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the audio features (input_features)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the labels (text tokens)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored by loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# Add the full audio path
def add_audio_path(example):
    example["audio"] = f"cv-corpus-22.0-2025-06-20/pl-siema/clips/{example['path']}"
    return example


def main():
    # Explicitly define the schema to avoid type inference issues
    features = Features({
        "client_id": Value("string"),
        "path": Value("string"),
        "sentence_id": Value("string"),
        "sentence": Value("string"),
        "sentence_domain": Value("string"),  # Explicitly string, not inferred
        "up_votes": Value("int64"),
        "down_votes": Value("int64"),
        "age": Value("string"),
        "gender": Value("string"),
        "accents": Value("string"),
        "variant": Value("string"),
        "locale": Value("string"),
        "segment": Value("string"),
    })

    dataset = load_dataset(
        "csv",
        data_files="cv-corpus-22.0-2025-06-20/pl-siema/validated.tsv",
        delimiter="\t",
        split="train",
        features=features  # Use explicit schema
    )
    dataset = dataset.map(add_audio_path)

    # Load processor EARLY (before preparing dataset)
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="pl", task="transcribe")
    print("=" * 50)
    print("Processor loaded successfully!")
    print("=" * 50)

    print("=" * 50)
    print("About to load Whisper model...")
    print("=" * 50)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    print("=" * 50)
    print("Model loaded successfully!")
    print("=" * 50)

    max_length = model.config.max_length
    # Filter long sentences
    def is_valid_length(example):
        tokens = processor.tokenizer(example["sentence"]).input_ids
        return len(tokens) <= max_length
    
    print(f"Before filtering: {len(dataset)} samples")
    dataset = dataset.filter(is_valid_length)
    print(f"After filtering: {len(dataset)} samples")

    dataset = dataset.shuffle(seed=42)
    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=42,
        shuffle=True  
    )
    test_valid = train_test["test"].train_test_split(
        test_size=0.5,
        seed=42,
        shuffle=True
    )

    # After splitting, check the distribution
    for split_name, split_data in [("train", train_test["train"]),
                                    ("test", test_valid["test"]),
                                    ("valid", test_valid["train"])]:
        print(f"\n{split_name} split:")
        sentences = split_data["sentence"]
        for sent in set(sentences):
            count = sentences.count(sent)
            print(f"  {sent}: {count}")

    dataset_dict = {
        "train": train_test["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]
    }

    print(f"Train: {len(dataset_dict['train'])} samples")
    print(f"Validation: {len(dataset_dict['valid'])} samples")
    print(f"Test: {len(dataset_dict['test'])} samples")
    
    def prepare_dataset(batch):
        audio, sr = librosa.load(batch["audio"], sr=16000)
        batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    cols_to_remove = dataset_dict["train"].column_names
    for split in ["train", "valid", "test"]:
       dataset_dict[split] = dataset_dict[split].map(
            prepare_dataset,
            remove_columns=cols_to_remove)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/whisper-base-polish-siema",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        warmup_steps=5,
        num_train_epochs=15,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=5,
        fp16=False,  # Keep disabled to prevent NaN on MPS
        dataloader_num_workers=4,
        report_to="none",
        max_grad_norm=1.0,  # Keep gradient clipping for stability
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer
    )

    trainer.train()

    # Save final model with all configs
    final_model_path = "./models/whisper-base-polish-siema-final"
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()

