from collections import Counter
import pandas as pd

def count_sources(df: pd.DataFrame, column_name: str, save_as: str = None) -> pd.DataFrame:
    df = df.copy()
    if column_name not in df.columns:
        raise ValueError(f"Kolom '{column_name}' tidak ditemukan di DataFrame.")

    df['sources_list'] = df[column_name].fillna('').apply(
        lambda x: [s.strip() for s in x.strip("'").split(',') if s.strip()]
    )
    df['source_counter'] = df['sources_list'].apply(len)
    df.drop(columns=['sources_list', column_name], inplace=True)

    if save_as:
        if save_as.endswith('.csv'):
            df.to_csv(save_as, index=False)
        elif save_as.endswith('.xlsx'):
            df.to_excel(save_as, index=False)
        elif save_as.endswith('.parquet'):
            df.to_parquet(save_as, index=False)
        elif save_as.endswith('.json'):
            df.to_json(save_as, orient='records', lines=True)
        elif save_as.endswith('.pkl'):
            df.to_pickle(save_as)
        else:
            raise ValueError("Format file tidak didukung. Gunakan .csv, .xlsx, .json, .parquet, atau .pkl.")

    return df
