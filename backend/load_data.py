import pandas as pd
from s3bucket import load_latest_snapshot_df

def add_file_paths_to_df() -> pd.DataFrame:
    """Load the latest snapshot of data from S3 bucket and add file paths."""
    df = load_latest_snapshot_df()
    for each row in df
        load picture into memory
        make a colomn with local file path in df
    
    for index, row in df.iterrows():
        
    
    return df


