# Whisper Fine-tuner

> **Note**: After training, convert your model to transformers.js format using [transformersjs-converter](https://github.com/markusingvarsson/transformersjs-whisper-converter)

Fine-tune OpenAI Whisper models on Common Voice datasets.

## Prerequisites

- Python 3.10+
- GPU recommended (CUDA or MPS)
- Common Voice dataset structure (download from Mozilla https://commonvoice.mozilla.org/en/datasets)

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Dataset Structure

Place Common Voice data in project root:

```
cv-corpus-22.0-2025-06-20/
  pl-siema/
    validated.tsv
    clips/
      *.mp3
```

Adjust paths in `src/whisper_finetuner/train.py` if using different dataset/language.

## Run Training

```bash
finetune-whisper
```

Training params (learning rate, epochs, batch size) are in `train.py:156-171`.

## Output

Model saved to `./models/whisper-base-polish-siema/`

## Notes

- Use Kaggle (30h/week free) or Colab Pro
- Current config: Polish language, 15 epochs, batch size 8
