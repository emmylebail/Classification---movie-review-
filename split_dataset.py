"""
On divise le dataset en trois, train, validation et test
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('IMDB_Dataset.csv')
print(f'Il y a au total {len(df)} données')
print('répartition de genres :')
print(df['sentiment'].value_counts())

# train (70%) / temporaire (30%) 
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42, # Control the shuffling applied to the data for reproducible output across multiple function calls
    stratify=df['sentiment']   # garder un équilibre entre les deux groupes de sentiment
)

# temporaire -- validation (15%) / test (15%) 
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df['sentiment']
)


train_df.to_csv('imdb_train.csv',      index=False)
val_df.to_csv('imdb_validation.csv',   index=False)
test_df.to_csv('imdb_test.csv',        index=False)
