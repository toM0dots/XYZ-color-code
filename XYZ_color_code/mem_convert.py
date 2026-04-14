import pandas as pd
import numpy as np
from benchmark import *

df = pd.read_csv("/Users/tomarr626/Downloads/ThermalTrails/full_error_history.csv")

# optional: keep rows ordered
df = df.sort_values(["T", "w", "h", "seed", "event_id"]).reset_index(drop=True)

w = int(df.iloc[0]["w"])
h = int(df.iloc[0]["h"])
n = w * h

all_A_Z = np.array([
    [int(c) for c in state.split("|")[0]]
    for state in df["state"]
], dtype=int)