# Named Entity Recognition (NER) with BERT

## 📌 Overview
This repository hosts the quantized version of the `bert-base-cased` model for Named Entity Recognition (NER) using the CoNLL-2003 dataset. The model is specifically designed to recognize entities related to **Person (PER), Organization (ORG), and Location (LOC)**. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## 🏗 Model Details
- **Model Architecture**: BERT Base Cased
- **Task**: Named Entity Recognition (NER)
- **Dataset**: Hugging Face's CoNLL-2003
- **Quantization**: BrainFloat16
- **Fine-tuning Framework**: Hugging Face Transformers

---
## 🚀 Usage

### Installation
```bash
pip install transformers torch
```

### Loading the Model
```python
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/bert-named-entity-recognition"
model = BertForTokenClassification.from_pretrained(model_name).to(device)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
```

### Named Entity Recognition Inference
```python
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
```
### **🔹 Labeling Scheme (BIO Format)**
 
- **B-XYZ (Beginning)**: Indicates the beginning of an entity of type XYZ (e.g., B-PER for the beginning of a person’s name).
- **I-XYZ (Inside)**: Represents subsequent tokens inside an entity (e.g., I-PER for the second part of a person’s name).
- **O (Outside)**: Denotes tokens that are not part of any named entity.

```
def predict_entities(text, model):

    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    tokens = {key: val.to(device) for key, val in tokens.items()}  # Move to CUDA

    with torch.no_grad():
        outputs = model(**tokens)
    
    logits = outputs.logits  # Extract logits
    predictions = torch.argmax(logits, dim=2)  # Get highest probability labels

    tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    predicted_labels = [label_list[pred] for pred in predictions[0].cpu().numpy()]

    final_tokens = []
    final_labels = []
    for token, label in zip(tokens_list, predicted_labels):
        if token.startswith("##"):  
            final_tokens[-1] += token[2:]  # Merge subword
        else:
            final_tokens.append(token)
            final_labels.append(label)

    for token, label in zip(final_tokens, final_labels):
        if token not in ["[CLS]", "[SEP]"]:
            print(f"{token}: {label}")

# 🔍 Test Example
sample_text = "Elon Musk is the CEO of Tesla, which is based in California."
predict_entities(sample_text, model)
```
---
## 📊 Evaluation Results for Quantized Model
 
### **🔹 Overall Performance**
 
- **Accuracy**: **97.10%** ✅
- **Precision**: **89.52%**
- **Recall**: **90.67%**
- **F1 Score**: **90.09%**
 
---
 
### **🔹 Performance by Entity Type**
 
| Entity Type | Precision | Recall | F1 Score | Number of Entities |
|------------|-----------|--------|----------|--------------------|
| **LOC** (Location) | **91.46%** | **92.07%** | **91.76%** | 3,000 |
| **MISC** (Miscellaneous) | **71.25%** | **72.83%** | **72.03%** | 1,266 |
| **ORG** (Organization) | **89.83%** | **93.02%** | **91.40%** | 3,524 |
| **PER** (Person) | **95.16%** | **94.04%** | **94.60%** | 2,989 |

 ---
#### ⏳ **Inference Speed Metrics**
- **Total Evaluation Time**: 15.89 sec
- **Samples Processed per Second**: 217.26
- **Steps per Second**: 27.18
- **Epochs Completed**: 3
 
---
## Fine-Tuning Details
### Dataset
The Hugging Face's `CoNLL-2003` dataset was used, containing texts and their ner tags.

## 📊 Training Details
- **Number of epochs**: 3  
- **Batch size**: 8  
- **Evaluation strategy**: epoch
- **Learning Rate**: 2e-5

### ⚡ Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

---
## 📂 Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

---
## ⚠️ Limitations
- The model may not generalize well to domains outside the fine-tuning dataset.
- Quantization may result in minor accuracy degradation compared to full-precision models.

---
## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
