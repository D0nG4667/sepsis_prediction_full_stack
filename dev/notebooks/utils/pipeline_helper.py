import pandas as pd
from typing import Union

numerical_features = ['prg', 'pl', 'pr', 'sk', 'ts', 'm11', 'bd2', 'age']

categorical_features = ['insurance']

new_features = ['age_group']

def as_category(data: Union[pd.DataFrame | pd.Series]) -> Union[pd.DataFrame | pd.Series]:
    return data.astype('category')

def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    if 'age_group' not in df_copy.columns and 'age' in df_copy.columns:
        df_copy['age_group'] = df_copy['age'].apply(lambda x: '60 and above' if x >= 60 else 'below 60')
        df_copy['age_group'] = as_category(df_copy['age_group'])
        df_copy.drop(columns='age', inplace=True)
        
    return df_copy


