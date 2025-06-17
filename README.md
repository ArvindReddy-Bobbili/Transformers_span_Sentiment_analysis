# Fine-Grained Sentiment Analysis: Span Extraction Project

## Overview

This project implements **fine-grained sentiment analysis** using span extraction techniques on product review data. The goal is to automatically identify and extract specific sentiment-bearing spans (phrases or words) from product reviews that correspond to particular aspects (like "spacebar" or "keys" for keyboard reviews).

### Problem Statement

Traditional sentiment analysis classifies entire reviews as positive/negative, but this project focuses on **aspect-based sentiment analysis** where we:

1. Identify specific product aspects mentioned in reviews
2. Extract the exact text spans that express sentiment about those aspects
3. Enable more granular understanding of what customers like/dislike about specific product features

## Models Implemented

This project compares three state-of-the-art transformer models for the span extraction task:

- **T5 (Text-to-Text Transfer Transformer)** - Google's versatile seq2seq model
- **BART (Bidirectional Auto-Regressive Transformers)** - Facebook's denoising autoencoder
- **Pegasus** - Google's model originally designed for abstractive summarization

## Dataset

The project uses product review data with two main components:

### Labeled Data (`labeled_data.csv`)

- **3,584 samples** of product reviews
- **Columns:**
  - `review`: Full product review text
  - `span`: Target sentiment span to extract
  - `Aspect`: Product aspect category (e.g., "spacebar", "keys")

### Unlabeled Data (`unlabeled_data.csv`)

- **8,833 samples** for inference
- Used to test trained models on new review data

### Example Data Structure

```
Review: "The spacebar is awful and makes typing difficult"
Span: "spacebar is awful"
Aspect: "spacebar"
```

## Project Structure

```
ML_Project_Span_Extraction/
│
├── data_outputs/
│   ├── labeled_data.csv           # Training data
│   ├── unlabeled_data.csv         # Test data
│   ├── output_BART_BASE.csv       # BART model predictions
│   ├── output_pegasus_1.csv       # Pegasus model predictions
│   ├── labeled.jpeg               # Data visualization
│   └── unlabeled.jpeg             # Data visualization
│
├── models/
│   ├── Fine_Grained_Sentinemt_Analysis_T5.ipynb      # T5 implementation
│   ├── Fine_Grained_Sentinemt_Analysis_BART.ipynb    # BART implementation
│   └── Fine_Grained_Sentinemt_Analysis_Pegasus.ipynb # Pegasus implementation
│
├── ML_Project_Report.pdf          # Detailed project report
└── README.md                      # This file
```

## Installation & Setup

### Requirements

- Python 3.7+
- PyTorch
- 🤗 Transformers library
- Google Colab (recommended for GPU access)

### Installation Commands

Run these commands in your notebook's first cell:

```bash
!pip install transformers[torch] datasets evaluate sacrebleu accelerate
```

### Additional Python Dependencies

```python
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import torch
import evaluate
```

## Usage Instructions

### 1. Data Preparation

1. Download the `labeled_data.csv` and `unlabeled_data.csv` files
2. Place them in the same directory as your notebook
3. The notebooks will automatically load and preprocess the data

### 2. Model Training

Each notebook follows the same structure:

```python
# Load and preprocess data
df = pd.read_csv('labeled_data.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

# Initialize model and tokenizer
checkpoint = "model-name"  # e.g., "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Train the model
trainer.train()
```

### 3. Running Inference

```python
# Generate predictions for unlabeled data
test_df = pd.read_csv('unlabeled_data.csv')
translator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

preds = []
for review in test_df['review']:
    preds.append(translator(review)[0]['generated_text'])
```

### 4. Training Time

- **Expected training time**: 30-45 minutes with T4 GPU on Google Colab
- **Recommended**: Use GPU acceleration for faster training

## Model Performance & Evaluation

The project uses multiple evaluation metrics:

### Metrics

- **BLEU Score**: Measures n-gram overlap between predicted and target spans
- **Jaccard Similarity**: Intersection over union of word sets
- **Confidence Score**: Model prediction confidence

### Training Configuration

- **Learning Rate**: 1e-4
- **Batch Size**: 16 (per device)
- **Epochs**: 5-15 (varies by model)
- **Optimizer**: AdamW with weight decay (0.02)
- **Mixed Precision**: FP16 for faster training

## Example Results

### Input Review:

```
"The volume controls and extra function keys are nice, but I wish it wasn't so big and bulky"
```

### Expected Output:

```
"so big and bulky"  # Extracted negative sentiment span
```

## Key Features

1. **Multi-Model Comparison**: Implements three different transformer architectures
2. **Aspect-Based Analysis**: Focuses on specific product aspects rather than overall sentiment
3. **Span Extraction**: Precisely identifies sentiment-bearing text portions
4. **Evaluation Framework**: Comprehensive metrics for model comparison
5. **Scalable Pipeline**: Easy to extend to new domains or aspects

## Technical Implementation Details

### Data Preprocessing

- Reviews are prefixed with "find span in the sentence: " for T5
- Maximum input length: 1024 tokens
- Maximum output length: 128 tokens
- Train/test split: 80/20

### Model Architecture

- **Encoder-Decoder Framework**: All models use sequence-to-sequence architecture
- **Attention Mechanism**: Self-attention and cross-attention for context understanding
- **Fine-tuning Strategy**: Task-specific fine-tuning on domain data

## Future Improvements

1. **Multi-Aspect Extraction**: Extract spans for multiple aspects simultaneously
2. **Sentiment Polarity**: Add positive/negative classification to extracted spans
3. **Domain Adaptation**: Extend to other product categories beyond keyboards
4. **Real-time Inference**: Deploy models for live review analysis
5. **Active Learning**: Incorporate human feedback to improve model performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## Citation

If you use this project in your research, please cite:

```
@misc{span_extraction_sentiment,
  title={Fine-Grained Sentiment Analysis via Span Extraction},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/your-repo/ML_Project_Span_Extraction}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- 🤗 Hugging Face for the Transformers library
- Google Colab for providing free GPU access
- Product review datasets for training data
