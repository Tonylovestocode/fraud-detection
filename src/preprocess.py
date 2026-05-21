from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent.parent / 'data' / 'creditcard.csv'


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Amount', 'Time'])

    X = df.drop(columns=['Class'])
    y = df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f'Train: {X_train.shape}  |  fraud cases: {y_train.sum()}')
    print(f'Test:  {X_test.shape}  |  fraud cases: {y_test.sum()}')
