import numpy as np
from pathlib import Path
import itertools
import sys

npz_file = Path(r'C:\Users\kahini\OneDrive\Desktop\SEM 7\RMS Course\cctv-multiview-summarization\data_storage\data\embeddings\view1_embeddings.npz')
if not npz_file.exists():
    print(f"Error: {npz_file} not found.")
    sys.exit(1)

print(f"Loading {npz_file}...")
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

# Subsample pairs to prevent memory/time explosion for full dataset
np.random.seed(42)

for tid, indices in track_groups.items():
    if len(indices) < 2: continue
    pairs = list(itertools.combinations(indices, 2))
    # sample at most 1000 pairs per track
    if len(pairs) > 1000:
        idx = np.random.choice(len(pairs), 1000, replace=False)
        pairs = [pairs[i] for i in idx]
    for i, j in pairs:
        same_pairs.append(np.dot(embs[i], embs[j]))

tids_list = list(track_groups.keys())
# sample across tracks to keep diff pairs reasonable
num_tracks = len(tids_list)
diff_samples = 0
for i in range(10000):
    t1, t2 = np.random.choice(tids_list, 2, replace=False)
    idx1 = np.random.choice(track_groups[t1])
    idx2 = np.random.choice(track_groups[t2])
    diff_pairs.append(np.dot(embs[idx1], embs[idx2]))

if not same_pairs or not diff_pairs:
    print("Error: Could not form enough pairs for validation.")
    sys.exit(1)

same_mean = np.mean(same_pairs)
diff_mean = np.mean(diff_pairs)

print(f"Same-track ({len(same_pairs)} pairs): mean={same_mean:.4f}")
print(f"Diff-track ({len(diff_pairs)} pairs): mean={diff_mean:.4f}")

# Require a meaningful separation (e.g. > 0.001)
diff = same_mean - diff_mean
print(f"Separation: {diff:.4f}")

if diff < 0.001:
    print("Validation FAILED: No meaningful separation between same-person and diff-person similarities. Likely collapse.")
    sys.exit(1)

print("Validation PASSED: Meaningful separation detected.")
sys.exit(0)
