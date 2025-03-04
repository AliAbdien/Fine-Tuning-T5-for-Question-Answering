# 🏆 Fine-Tuning T5 for Question Answering

This repository contains the implementation for **fine-tuning T5** on a **Question Answering (QA)** dataset. The model leverages **Hugging Face's Transformers** library, utilizing **pre-trained T5 models** for better accuracy in answering questions.

## 📂 Project Structure

```
📦 Fine-Tuning-T5-QA
│── 📜 config.json          # Model configuration file
│── 📜 generation_config.json  # Text generation settings
│── 📜 special_tokens_map.json  # Special tokens configuration
│── 📜 spiece.model         # SentencePiece tokenizer model
│── 📜 tokenizer.json       # Tokenizer vocabulary and merges
│── 📜 tokenizer_config.json  # Tokenizer settings
│── 📜 T5_QA.ipynb          # Jupyter Notebook for fine-tuning and inference
```

## 📌 Features

✅ Fine-tunes **T5** for **extractive & abstractive QA**  
✅ Implements **custom tokenization** using SentencePiece  
✅ Leverages **Hugging Face Transformers** for training & inference  
✅ Supports **structured model checkpoints**  
✅ Optimized for **GPU acceleration** using PyTorch & Safetensors  

## 🛠 Installation

Ensure you have Python 3.8+ installed and install the required dependencies:

```bash
pip install torch transformers sentencepiece safetensors
```

## 🚀 Fine-Tuning T5 for QA

You can fine-tune the model using the provided Jupyter Notebook:

```bash
jupyter notebook T5_QA.ipynb
```

Alternatively, execute the following script to load and train the model:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example input
input_text = "question: What is AI? context: Artificial Intelligence (AI) is the simulation of human intelligence in machines."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate an answer
output_ids = model.generate(input_ids, max_length=50)
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Answer:", answer)
```

## 📊 Generating Answers with Fine-Tuned Model

Once the model is trained, you can use it to answer questions:

```python
from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="./model.safetensors", tokenizer=tokenizer)
output = qa_pipeline("question: What is machine learning? context: Machine learning is a subset of AI that focuses on learning from data.")
print(output)
```


---

📧 **Contact:** ali.abdien.omar@gmail.com  

---
