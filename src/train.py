import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'model.pkl'


def train():
    print('Loading and preprocessing data...')
    X_train, _, y_train, _ = load_and_preprocess()
    print(f'Train set: {X_train.shape[0]:,} rows  |  fraud: {y_train.sum()}  |  legit: {(y_train == 0).sum():,}')

    print('Training Random Forest (class_weight=balanced)...')
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {MODEL_PATH}')


if __name__ == '__main__':
    train()
