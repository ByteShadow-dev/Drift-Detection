import pandas as pd
import numpy as np
df = pd.DataFrame({'userId': [1, 1, 2, 2], 'kl_score': [0.1, 0.2, 0.5, 0.6]})
def apply_fn(chunk):
    chunk = chunk.copy()
    chunk['mean'] = chunk['kl_score'].mean()
    return chunk
res = df.groupby('userId', include_groups=False).apply(apply_fn).reset_index(level=0).reset_index(drop=True)
print(res.columns)
print(res)
