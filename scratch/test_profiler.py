import pandas as pd
import sys
import os

# Ensure local import paths work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Aura")))

from utils.profiler import profile_dataset

# Create a mock dataframe with floats
df = pd.DataFrame({
    "Class": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    "V1": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1]
})

profile = profile_dataset(df)
print("Profile outcome for 'Class':", profile["columns"]["Class"])
