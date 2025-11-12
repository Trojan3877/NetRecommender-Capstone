from pathlib import Path
import pandas as pd

def preprocess_data():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    # TODO: replace with real ETL (e.g., MovieLens)
    df = pd.DataFrame({"user_id":[1,1,2], "item_id":[10,11,10], "rating":[5,4,3]})
    df.to_csv("data/processed/train.csv", index=False)
    print("✅ Saved processed data.")
