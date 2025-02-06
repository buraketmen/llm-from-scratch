# LLM-FROM-SCRATCH

Created to learn how to build a LLM from scratch. Tiktoken used for tokenization, and PyTorch for the model.

## Installation

```
git clone https://github.com/buraketmen/llm-from-scratch.git
cd llm-from-scratch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python ./llm/main.py --train # Train
python ./llm/main.py --generate # Generate text with default text
python ./llm/main.py --generate --prompt "Khaleesi and Jon Snow were walking " # Generate text with prompt
```
