# Fraud Detection

A machine-learning project for detecting fraudulent transactions. It covers the
full workflow — exploratory analysis, preprocessing, model training, and
evaluation — with a focus on the class-imbalance problem that makes fraud
detection hard (fraudulent transactions are rare compared to legitimate ones).

## Project Structure

```
fraud-detection/
├── notebooks/
│   └── exploration.ipynb   # Exploratory data analysis
├── src/
│   ├── preprocess.py       # Data cleaning & feature preparation
│   ├── train.py            # Model training
│   └── evaluate.py         # Model evaluation & metrics
└── requirements.txt
```

## Tech Stack

- **Python**, **pandas**, **NumPy**
- **scikit-learn** for modeling
- **imbalanced-learn** for handling class imbalance (e.g. SMOTE)
- **matplotlib** / **seaborn** for visualization
- **Jupyter** for exploration

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Or explore interactively
jupyter notebook notebooks/exploration.ipynb
```

## Notes

Because fraud is a highly imbalanced classification problem, evaluation focuses
on precision, recall, and F1 for the fraud class rather than raw accuracy.
