import numpy as np
from pathlib import Path

data_dir = Path(r'C:\Users\kahini\OneDrive\Desktop\SEM 7\RMS Course\cctv-multiview-summarization\data_storage\data\embeddings')
for view in ['view1', 'view2', 'view3']:
    npz = data_dir / f'{view}_embeddings.npz'
    if npz.exists():
        with np.load(npz) as data:
            embs = data['embeddings']
            if len(embs) >= 2:
                sim1 = np.dot(embs[0], embs[1])
                sim2 = np.dot(embs[0], embs[-1])
                print(f'{view} (0 and 1): {sim1:.4f}, (0 and -1): {sim2:.4f}')
    else:
        print(f'{view}: File not found')
