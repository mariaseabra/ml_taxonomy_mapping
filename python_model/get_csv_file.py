import pandas as pd

def parse_csv_file(file_path):
    return pd.read_csv(file_path).to_dict('records')
