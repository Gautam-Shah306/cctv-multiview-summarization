"""
stitch_tracklets.py

Stage 1c script to heal broken tracklets (fragmentation) prior to cross-view clustering.
Identifies tracklets in the same view that are adjacent in time and visually identical,
and stitches them into single unified tracklets.

Guards against over-merging by enforcing a hard maximum span limit.

Outputs:
- data_manifests/tracklet_features_stitched.csv
- data/tracklet_embeddings_stitched.npz
"""

# Group 1 — standard library
import argparse
import csv
from pathlib import Path

# Group 2 — third-party
import numpy as np

# Group 3 — local / project
from config import DATA_DIR

# --- Constants ----------------------------------------------------------------
MAX_GAP_FRAMES = 15
SIMILARITY_THRESHOLD = 0.85
# Hard cap on how long a stitched tracklet can be to prevent runaway transitive
# merging across huge time gaps (e.g. 750 frames = 30 seconds).
MAX_STITCHED_SPAN_FRAMES = 750

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data_manifests" / "tracklet_features.csv"
OUT_MANIFEST = Path(__file__).resolve().parent.parent / "data_manifests" / "tracklet_features_stitched.csv"
IN_NPZ_PATH = DATA_DIR / "tracklet_embeddings.npz"
OUT_NPZ_PATH = DATA_DIR / "tracklet_embeddings_stitched.npz"


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector to unit L2 norm."""
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        return (vector / norm).astype(np.float32)
    return vector.astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two unit L2-normalized vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite outputs if they exist")
    args = parser.parse_args()

    if OUT_MANIFEST.exists() and OUT_NPZ_PATH.exists() and not args.force:
        print(f"[SKIP] Stitched outputs already exist at {OUT_MANIFEST.name} and {OUT_NPZ_PATH.name}. Use --force to re-run.")
        return

    # 1. Load Tracklet Features
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "view": r["view"],
                "track_id": int(r["track_id"]),
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "start_time": float(r["start_time"]),
                "end_time": float(r["end_time"]),
                "num_detections": int(r["num_detections"])
            })

    # Group by view and sort
    views = {}
    for r in rows:
        v = r["view"]
        if v not in views:
            views[v] = []
        r["used"] = False
        views[v].append(r)
        
    for v in views:
        views[v].sort(key=lambda x: x["start_frame"])

    # 2. Load Embeddings
    embs = {}
    with np.load(IN_NPZ_PATH) as data:
        for k in data.files:
            embs[k] = data[k]

    # 3. Stitch Tracklets
    stitched_rows = []
    stitched_embs = {}
    
    total_original = 0
    total_stitched = 0
    cap_hits = 0
    chain_sizes = {}

    print("=" * 60)
    print(f"[INFO] Stitching Tracklets (gap <= {MAX_GAP_FRAMES} frames, sim > {SIMILARITY_THRESHOLD})")
    print(f"[INFO] Span Cap: {MAX_STITCHED_SPAN_FRAMES} frames")
    print("=" * 60)

    for view_name, v_tracklets in views.items():
        original_count = len(v_tracklets)
        total_original += original_count
        stitched_for_view = 0
        
        # New stitched track IDs will just be sequential ints starting from 1
        next_stitched_id = 1

        for i, t in enumerate(v_tracklets):
            if t["used"]:
                continue
            
            chain = [t]
            t["used"] = True
            
            # Greedily grow the chain forward in time
            while True:
                tail = chain[-1]
                next_cand = None
                
                # Search forward from the tail's position
                for j in range(i + 1, len(v_tracklets)):
                    cand = v_tracklets[j]
                    if cand["used"]:
                        continue
                    
                    gap = cand["start_frame"] - tail["end_frame"]
                    
                    if gap < 0:
                        continue  # overlapping in time, cannot be the same person's tracklet reliably
                    if gap > MAX_GAP_FRAMES:
                        # Since tracklets are sorted by start_frame, subsequent candidates will only have larger gaps
                        break
                    
                    # Gap is valid. Check visual similarity
                    k_tail = f"{view_name}_{tail['track_id']}"
                    k_cand = f"{view_name}_{cand['track_id']}"
                    
                    if k_tail in embs and k_cand in embs:
                        sim = cosine_similarity(embs[k_tail], embs[k_cand])
                        if sim > SIMILARITY_THRESHOLD:
                            # Check hard span cap
                            span = cand["end_frame"] - chain[0]["start_frame"] + 1
                            if span > MAX_STITCHED_SPAN_FRAMES:
                                print(f"[WARNING] Prevented over-merge in {view_name}: Chain spanning frames "
                                      f"{chain[0]['start_frame']}-{tail['end_frame']} + track_id {cand['track_id']} "
                                      f"would exceed {MAX_STITCHED_SPAN_FRAMES} frames.")
                                cap_hits += 1
                                break # Stop growing this chain, leave the candidate for another chain
                            
                            next_cand = cand
                            break # Found the best adjacent candidate
                
                if next_cand is not None:
                    chain.append(next_cand)
                    next_cand["used"] = True
                else:
                    break # No more valid candidates found, finish this chain
            
            # Finalize the stitched chain
            c_len = len(chain)
            chain_sizes[c_len] = chain_sizes.get(c_len, 0) + 1
            
            start_frame = min(tr["start_frame"] for tr in chain)
            end_frame = max(tr["end_frame"] for tr in chain)
            start_time = min(tr["start_time"] for tr in chain)
            end_time = max(tr["end_time"] for tr in chain)
            num_dets = sum(tr["num_detections"] for tr in chain)
            orig_ids = [tr["track_id"] for tr in chain]
            
            # Pool embeddings
            raw_embs = []
            for tr in chain:
                k = f"{view_name}_{tr['track_id']}"
                if k in embs:
                    raw_embs.append(embs[k])
                    
            if raw_embs:
                mean_emb = np.mean(raw_embs, axis=0)
                pooled_emb = normalize_l2(mean_emb)
            else:
                pooled_emb = np.zeros(512, dtype=np.float32)
                
            out_key = f"{view_name}_{next_stitched_id}"
            stitched_embs[out_key] = pooled_emb
            
            stitched_rows.append({
                "view": view_name,
                "stitched_track_id": next_stitched_id,
                "original_track_ids": str(orig_ids),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": f"{start_time:.2f}",
                "end_time": f"{end_time:.2f}",
                "num_detections": num_dets
            })
            
            next_stitched_id += 1
            stitched_for_view += 1

        print(f"[INFO] {view_name}: {original_count} original -> {stitched_for_view} stitched tracklets")
        total_stitched += stitched_for_view

    # 4. Write Outputs
    fieldnames = [
        "view",
        "stitched_track_id",
        "original_track_ids",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "num_detections"
    ]
    
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stitched_rows)

    np.savez_compressed(OUT_NPZ_PATH, **stitched_embs)
    
    # 5. Final Report
    print("\n" + "=" * 60)
    print("TRACKLET STITCHING SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Original Tracklets: {total_original}")
    print(f"Total Stitched Tracklets: {total_stitched}")
    
    if total_original > 0:
        reduction = (total_original - total_stitched) / total_original * 100
        print(f"Reduction: {reduction:.1f}%")
        
    print(f"\nOver-merge Cap Hits (Span > {MAX_STITCHED_SPAN_FRAMES}): {cap_hits}")
    
    print("\nChain Size Distribution (Original Tracklets per Stitched Tracklet):")
    for size in sorted(chain_sizes.keys()):
        if size == 1:
            print(f"  - {chain_sizes[size]} stitched tracklets are 1 original tracklet unchanged")
        else:
            print(f"  - {chain_sizes[size]} stitched tracklets are {size} original tracklets merged")
    
    print("=" * 60)
    print(f"[DONE] Written to: {OUT_MANIFEST.name} and {OUT_NPZ_PATH.name}")


if __name__ == "__main__":
    main()
