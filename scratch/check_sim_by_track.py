import numpy as np
from pathlib import Path
import itertools

npz_file = Path(r'C:\Users\kahini\OneDrive\Desktop\SEM 7\RMS Course\cctv-multiview-summarization\data_storage\data\embeddings\view1_embeddings.npz')

with np.load(npz_file) as data:
    embs = data['embeddings']
    tids = data['track_ids']

track_groups = {}
for i, tid in enumerate(tids):
    if tid == -1: continue
    if tid not in track_groups: track_groups[tid] = []
    track_groups[tid].append(i)

same_pairs = []
diff_pairs = []

for tid, indices in track_groups.items():
    if len(indices) < 2: continue
    for i, j in itertools.combinations(indices, 2):
        same_pairs.append(np.dot(embs[i], embs[j]))

tids_list = list(track_groups.keys())
print(f"Unique valid track IDs in this subset: {tids_list}")

for idx1 in range(len(tids_list)):
    for idx2 in range(idx1 + 1, len(tids_list)):
        t1, t2 = tids_list[idx1], tids_list[idx2]
        for i in track_groups[t1]:
            for j in track_groups[t2]:
                diff_pairs.append(np.dot(embs[i], embs[j]))

if same_pairs:
    print(f"Same-track ({len(same_pairs)} pairs): min={min(same_pairs):.4f}, mean={np.mean(same_pairs):.4f}, max={max(same_pairs):.4f}")
if diff_pairs:
    print(f"Diff-track ({len(diff_pairs)} pairs): min={min(diff_pairs):.4f}, mean={np.mean(diff_pairs):.4f}, max={max(diff_pairs):.4f}")
