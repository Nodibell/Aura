from huggingface_hub import hf_hub_download
import json
import pandas as pd

repo_id = "FIdo-AI/ua-squad"
local_path = hf_hub_download(repo_id=repo_id, filename="train.json", repo_type="dataset")

with open(local_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data["data"])
print(df.head(3))
print("\nShape:", df.shape)
print("\nData types:")
print(df.dtypes)
