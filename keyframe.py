#!/usr/bin/env python3
"""
Keyframe / Screenshot Selector for Short Videos

Approach (no heavy ML dependencies by default):
  1) Sample frames at a target FPS.
  2) Detect shot boundaries via HSV histogram deltas.
  3) Score frames on sharpness, entropy (detail), edge density, and letterbox penalty.
  4) Select diverse, high‑scoring frames via k‑means on color histograms.

Optional (if you install torch + openai/CLIP):
  - Add semantic/aesthetic scoring; see the CLIP hook near the bottom.

Usage:
  python keyframe_select.py input.mp4 --num 5 --fps 2 --outdir frames_out

Dependencies:
  pip install opencv-python numpy scikit-learn

Notes:
  - OpenCV must be able to read your codec (ffmpeg backend). If it fails, transcode first:
      ffmpeg -i input.mov -c:v libx264 -crf 18 -preset veryfast -c:a copy input.mp4
"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_str(frame_idx: int, fps: float) -> str:
    t = frame_idx / max(fps, 1e-6)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    if h:
        return f"{h:02d}-{m:02d}-{s:05.2f}"
    return f"{m:02d}-{s:05.2f}"


def hsv_hist(img_bgr: np.ndarray, bins=(8, 12, 12)) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    # Smaller = more similar; convert to a distance-like score
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean()) / 255.0


def analyze_viewing_angle(img_bgr: np.ndarray) -> Tuple[str, float]:
    """Analyze viewing angle using geometric features rather than CLIP semantics."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Detect contours to analyze shape orientation
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "unknown", 0.0
    
    # Find largest contour (likely the main food item)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 1000:  # Too small to analyze
        return "unknown", 0.0
    
    # Analyze bounding box aspect ratio
    x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    
    # Analyze contour properties
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Fit ellipse to understand orientation
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        ellipse_ratio = major_axis / max(minor_axis, 1)
    else:
        ellipse_ratio = 1.0
        angle = 0
    
    # Heuristic classification based on geometric properties
    confidence = 0.0
    angle_type = "unknown"
    
    # Top-down view: more circular, higher circularity
    if circularity > 0.6 and ellipse_ratio < 1.5:
        angle_type = "top-down"
        confidence = circularity
    
    # Side view: elongated horizontally, lower circularity
    elif aspect_ratio > 1.8 and circularity < 0.4:
        angle_type = "side"
        confidence = aspect_ratio / 3.0  # normalize
    
    # Front view: more square-ish
    elif 0.7 < aspect_ratio < 1.4 and circularity < 0.6:
        angle_type = "front"
        confidence = 1.0 - abs(aspect_ratio - 1.0)
    
    # Diagonal: moderate elongation, moderate circularity
    else:
        angle_type = "diagonal"
        confidence = 0.5
    
    return angle_type, min(confidence, 1.0)


def letterbox_penalty(img_bgr: np.ndarray, thresh: int = 16) -> float:
    # Penalize thick black/constant bars at top/bottom or sides
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # sample edge rows/cols
    top = gray[0: int(0.06 * h), :]
    bot = gray[h - int(0.06 * h):, :]
    left = gray[:, 0: int(0.06 * w)]
    right = gray[:, w - int(0.06 * w):]
    def row_pen(mat):
        return float((mat.std() < thresh) * 1.0)
    # average of four edges
    return (row_pen(top) + row_pen(bot) + row_pen(left) + row_pen(right)) / 4.0


# -----------------------------
# Shot detection
# -----------------------------

def detect_shots(frames: List[np.ndarray], hist_bins=(8, 12, 12), hard_cut_thresh=0.5) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive for each shot."""
    if not frames:
        return []
    hists = [hsv_hist(f, bins=hist_bins) for f in frames]
    cuts = [0]
    for i in range(1, len(hists)):
        d = bhattacharyya(hists[i-1], hists[i])
        if d > hard_cut_thresh:
            cuts.append(i)
    cuts.append(len(frames))
    shots = []
    for i in range(len(cuts)-1):
        s, e = cuts[i], cuts[i+1]-1
        if s <= e:
            shots.append((s, e))
    return shots

# -----------------------------
# Frame scoring
# -----------------------------

@dataclass
class FrameScore:
    idx: int
    score: float
    components: Tuple[float, float, float, float, float]  # Added CLIP score component


def score_frames(frames: List[np.ndarray], outdir: str = "keyframes_out") -> List[FrameScore]:
    if not frames:
        return []
    gray_list = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    sharp = np.array([variance_of_laplacian(g) for g in gray_list], dtype=float)
    ent = np.array([shannon_entropy(g) for g in gray_list], dtype=float)
    edge = np.array([edge_density(g) for g in gray_list], dtype=float)
    penal = np.array([letterbox_penalty(frames[i]) for i in range(len(frames))], dtype=float)
    
    # Add CLIP scoring for semantic/aesthetic quality
    clip_scores = clip_score_batched(frames, outdir)

    # Normalize to 0..1
    def norm(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(x.max(), x.min()):
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    sharp_n = norm(sharp)
    ent_n = norm(ent)
    edge_n = norm(edge)
    penal_n = penal  # already 0..1
    clip_n = clip_scores  # already normalized 0..1

    # Weighted sum; CLIP gets high weight for semantic quality
    w_sharp, w_ent, w_edge, w_penal, w_clip = 0.25, 0.20, 0.15, 0.15, 0.40
    raw = w_sharp * sharp_n + w_ent * ent_n + w_edge * edge_n - w_penal * penal_n + w_clip * clip_n

    return [FrameScore(i, float(raw[i]), (float(sharp_n[i]), float(ent_n[i]), float(edge_n[i]), float(penal_n[i]), float(clip_n[i]))) for i in range(len(frames))]

# -----------------------------
# Diversity selection via k-means on color histograms
# -----------------------------

def select_diverse_topk(frames: List[np.ndarray], scores: List[FrameScore], k: int) -> List[int]:
    if not frames or not scores:
        return []
    k = max(1, min(k, len(frames)))

    if len(frames) <= k:
        return [fs.idx for fs in sorted(scores, key=lambda s: s.score, reverse=True)]

    # Use CLIP embeddings for angle/perspective diversity instead of color histograms
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    
    with torch.no_grad():
        img_feat = model.encode_image(batch)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        feats = img_feat.float().cpu().numpy()

    # To make k-means stable across runs
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(feats)

    # For each cluster, pick the highest-scoring frame
    best_per_cluster = {}
    for fs in scores:
        c = int(labels[fs.idx])
        if c not in best_per_cluster or fs.score > best_per_cluster[c].score:
            best_per_cluster[c] = fs

    chosen = sorted(best_per_cluster.values(), key=lambda s: s.score, reverse=True)
    return [fs.idx for fs in chosen]

# -----------------------------
# Video sampling
# -----------------------------

def sample_video(path: str, target_fps: float, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float, List[int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps))) if target_fps > 0 else 1

    frames = []
    frame_indices = []
    idx = 0
    taken = 0

    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            frames.append(frame)
            frame_indices.append(idx)
            taken += 1
            if max_frames and taken >= max_frames:
                break
        idx += 1

    cap.release()
    return frames, src_fps, frame_indices

# -----------------------------
# Main pipeline
# -----------------------------

def pick_keyframes(video_path: str, outdir: str, num: int = 5, fps: float = 2.0,
                   hard_cut_thresh: float = 0.5, min_gap_seconds: float = 1.0,
                   hist_bins=(8, 12, 12)) -> List[Tuple[str, int, float]]:
    """
    Returns list of (output_path, source_frame_index, source_fps)
    """
    ensure_dir(outdir)

    frames, src_fps, src_indices = sample_video(video_path, fps)
    if not frames:
        raise RuntimeError("No frames sampled; check codec/FPS settings.")

    # Shot detection on sampled frames
    shots = detect_shots(frames, hist_bins=hist_bins, hard_cut_thresh=hard_cut_thresh)

    # Score per frame within each shot, keep the top few per shot to avoid biasing to long shots
    all_candidates: List[FrameScore] = []
    per_shot_keep = max(1, int(math.ceil(num * 2 / max(1, len(shots)))))  # over-generate a bit

    frame_scores = score_frames(frames, outdir)

    for (s, e) in shots:
        shot_scores = sorted(frame_scores[s:e+1], key=lambda fs: fs.score, reverse=True)
        all_candidates.extend(shot_scores[:per_shot_keep])

    # Sort global candidates and apply diversity with k-means
    all_candidates = sorted(all_candidates, key=lambda fs: fs.score, reverse=True)

    # Enforce a minimal temporal gap between picks (on sampled grid)
    min_gap_frames = int(round(min_gap_seconds * fps))

    # Diversity selection on top M candidates
    topM = min(len(frames), max(num * 4, num + 3))
    cand_indices = [fs.idx for fs in all_candidates[:topM]]

    # Create remapped scores for diverse selection
    cand_frames = [frames[i] for i in cand_indices]
    cand_scores = []
    for i, orig_idx in enumerate(cand_indices):
        cand_scores.append(FrameScore(i, frame_scores[orig_idx].score, frame_scores[orig_idx].components))
    
    diverse_subset_indices = select_diverse_topk(cand_frames, cand_scores, k=num * 2)
    # Map back to original frame indices
    diverse_indices = [cand_indices[i] for i in diverse_subset_indices]

    # Now pick greedily with temporal spacing
    picked = []
    for i in sorted(diverse_indices, key=lambda j: frame_scores[j].score, reverse=True):
        if all(abs(i - p) >= min_gap_frames for p in picked):
            picked.append(i)
            if len(picked) >= num:
                break

    # Fallback if not enough picked
    if len(picked) < num:
        for i in [fs.idx for fs in all_candidates]:
            if i not in picked and all(abs(i - p) >= min_gap_frames for p in picked):
                picked.append(i)
                if len(picked) >= num:
                    break

    results = []
    print("\nSelected keyframes with geometric angle analysis:")
    for i in sorted(picked):
        frame = frames[i]
        
        # Analyze viewing angle using geometric features
        angle_type, confidence = analyze_viewing_angle(frame)
        
        # Write file with original timestamp (approx via source indices & fps ratio)
        src_idx = src_indices[i]
        stamp = ts_str(src_idx, src_fps)
        
        # Extract video name without extension
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        out_path = os.path.join(outdir, f"{video_basename}_{stamp}_{angle_type}.png")
        cv2.imwrite(out_path, frame)
        results.append((out_path, int(src_idx), float(src_fps)))
        
        print(f"  {stamp}: {angle_type} view (confidence: {confidence:.3f})")

    return results

# -----------------------------
# CLIP scoring for semantic/aesthetic quality (enabled)
# -----------------------------
import torch
import clip
from PIL import Image

def clip_score_batched(frames: List[np.ndarray], outdir: str = "keyframes_out", device: str = "mps" if torch.backends.mps.is_available() else "cpu") -> np.ndarray:
    """Score frames using CLIP for diverse viewing angles of food items. Uses MPS on M-series Macs for faster processing."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Pasta-specific viewing angles for portion/calorie estimation
    angle_prompts = [
        "viewed from directly above, overhead shot, bird's eye view showing full plate coverage and portion size",
        "photographed from the side showing pile height, thickness, and vertical dimension of the serving",
        "shot from front angle showing width and spread of the portion on the plate",
        "at diagonal angle showing 3D volume, depth, and how much pasta is stacked or layered",
        "close-up pasta showing individual noodle thickness and density for portion estimation"
    ]
    
    texts = clip.tokenize(angle_prompts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(texts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    
    with torch.no_grad():
        img_feat = model.encode_image(batch)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        # Get similarity to all angle prompts
        sims = (img_feat @ text_feat.T).float().cpu().numpy()  # [num_frames, num_prompts]
        
        # Instead of just taking max, let's return detailed angle analysis
        angle_names = ["overhead", "side", "front", "diagonal", "close-up"]
        
        print("Frame angle analysis:")
        for i, frame_sims in enumerate(sims):
            best_angle_idx = frame_sims.argmax()
            best_angle = angle_names[best_angle_idx]
            confidence = frame_sims[best_angle_idx]
            print(f"  Frame {i}: Best angle = {best_angle} (confidence: {confidence:.3f})")
        
        # Analyze principal components of viewing angle variations
        print("\nPrincipal Component Analysis of Viewing Angles:")
        pca = PCA(n_components=3)
        angle_pca = pca.fit_transform(sims)  # sims is [num_frames, num_angle_prompts]
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Principal components (angle space):")
        for i, component in enumerate(pca.components_):
            dominant_angles = []
            for j, weight in enumerate(component):
                if abs(weight) > 0.3:  # threshold for significant contribution
                    dominant_angles.append(f"{angle_names[j]}({weight:.2f})")
            print(f"  PC{i+1}: {', '.join(dominant_angles) if dominant_angles else 'mixed'}")
        
        # Find frames with maximum variation in each principal component
        print(f"\nFrames with maximum variation:")
        for i in range(min(3, angle_pca.shape[1])):
            max_idx = np.argmax(np.abs(angle_pca[:, i]))
            print(f"  PC{i+1} max variation: Frame {max_idx} (value: {angle_pca[max_idx, i]:.3f})")
        
        # Create 3D visualization of the viewing angle space
        visualize_angle_space(sims, angle_names, outdir)
            
        # Take max similarity across all angle prompts for each frame
        max_sims = sims.max(axis=1)
    
    # normalize 0..1
    max_sims = (max_sims - max_sims.min()) / (max_sims.max() - max_sims.min() + 1e-9)
    return max_sims


def visualize_angle_space(sims: np.ndarray, angle_names: List[str], outdir: str):
    """Create 3D visualization of viewing angle space with principal component vectors."""
    pca = PCA(n_components=3)
    angle_pca = pca.fit_transform(sims)
    
    fig = plt.figure(figsize=(12, 4))
    
    # 3D scatter plot of frames in angle space
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(angle_pca[:, 0], angle_pca[:, 1], angle_pca[:, 2], 
                         c=range(len(angle_pca)), cmap='viridis', s=50)
    
    # Draw principal component vectors from origin
    origin = np.mean(angle_pca, axis=0)
    scale = np.std(angle_pca, axis=0) * 2
    
    for i, (component, color) in enumerate(zip(pca.components_, ['red', 'green', 'blue'])):
        # Project component back to 3D PCA space
        vec_end = origin + component[:3] * scale[i] if len(component) >= 3 else origin
        ax1.quiver(origin[0], origin[1], origin[2], 
                  vec_end[0]-origin[0], vec_end[1]-origin[1], vec_end[2]-origin[2],
                  color=color, arrow_length_ratio=0.1, linewidth=3, 
                  label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.2f})')
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2') 
    ax1.set_zlabel('PC3')
    ax1.set_title('3D Viewing Angle Space')
    ax1.legend()
    
    # 2D projections
    ax2 = fig.add_subplot(132)
    ax2.scatter(angle_pca[:, 0], angle_pca[:, 1], c=range(len(angle_pca)), cmap='viridis')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PC1 vs PC2')
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(angle_pca[:, 0], angle_pca[:, 2], c=range(len(angle_pca)), cmap='viridis')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC3')
    ax3.set_title('PC1 vs PC3')
    
    plt.tight_layout()
    viz_path = os.path.join(outdir, 'angle_space_analysis.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved viewing angle space visualization to: {viz_path}")
    return viz_path

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Pick diverse, high-quality keyframes from a short video")
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument("--outdir", default="keyframes_out", help="Directory to save selected frames")
    ap.add_argument("--num", type=int, default=5, help="Number of keyframes to extract")
    ap.add_argument("--fps", type=float, default=2.0, help="Sampling FPS for analysis")
    ap.add_argument("--hard_cut", type=float, default=0.5, help="Bhattacharyya distance threshold for shot cuts (higher = fewer cuts)")
    ap.add_argument("--min_gap_seconds", type=float, default=1.0, help="Min seconds between chosen frames (on sampled grid)")
    args = ap.parse_args()

    results = pick_keyframes(args.video, args.outdir, num=args.num, fps=args.fps,
                             hard_cut_thresh=args.hard_cut, min_gap_seconds=args.min_gap_seconds)
    print("Saved frames:")
    for path, src_idx, src_fps in results:
        print(f"  {path} (frame {src_idx} @ ~{src_fps:.2f} fps)")

if __name__ == "__main__":
    main()
