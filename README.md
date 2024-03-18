# SMART: Automatically Scaling Down Language Models with Accuracy Guarantees for Reduced Processing Fees

> SMART, Scaling Models Adaptively for Reduced Token Fees, is a novel LLM framework designed to minimize the inference costs of NLP tasks while ensuring sufficient result quality.


## Quick Start

```bash
# Tested on Python 3.10.
git clone https://github.com/saehanjo/smart-llms.git
cd smart-llms

# (Optional) Create virtual environment.
python -m venv .venv
source .venv/bin/activate

# Install requirements.
pip install -r requirements.txt

# Run experiments. Results are saved in the CSV file: result/all.csv.
mkdir result
python run_exp.py
```