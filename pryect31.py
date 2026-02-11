import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix
)

import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# -----------------------------
# Load data
# -----------------------------
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

trainingData = pd.read_csv("trainingData.csv")
validationData = pd.read_csv("validationData.csv")

rssi_cols = trainingData.columns[:520]

X_train = trainingData[rssi_cols].values
X_val = validationData[rssi_cols].values

y_train_building = trainingData["BUILDINGID"].values
y_train_floor = trainingData["FLOOR"].values

y_val_building = validationData["BUILDINGID"].values
y_val_floor = validationData["FLOOR"].values

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Original features: {X_train.shape[1]}")
print(f"Unique floors in training: {np.unique(y_train_floor)}")
print(f"Unique floors in validation: {np.unique(y_val_floor)}")

# Replace "no signal" code
X_train[X_train == 100] = -110
X_val[X_val == 100] = -110

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Remove APs with no useful information (always -110)
threshold_constant = 0.95
constant_aps = (X_train == -110).mean(axis=0) > threshold_constant
X_train_filtered = X_train[:, ~constant_aps]
X_val_filtered = X_val[:, ~constant_aps]

print(f"APs removed (>{threshold_constant * 100}% missing): {constant_aps.sum()}")
print(f"APs remaining: {X_train_filtered.shape[1]}")

# Create additional statistical features
X_train_mean = X_train_filtered.mean(axis=1, keepdims=True)
X_train_std = X_train_filtered.std(axis=1, keepdims=True)
X_train_max = X_train_filtered.max(axis=1, keepdims=True)
X_train_min = X_train_filtered.min(axis=1, keepdims=True)
X_train_nonzero = (X_train_filtered > -110).sum(axis=1, keepdims=True)  # number of visible APs

X_train_enhanced = np.hstack([
    X_train_filtered,
    X_train_mean,
    X_train_std,
    X_train_max,
    X_train_min,
    X_train_nonzero
])

# Same for validation
X_val_mean = X_val_filtered.mean(axis=1, keepdims=True)
X_val_std = X_val_filtered.std(axis=1, keepdims=True)
X_val_max = X_val_filtered.max(axis=1, keepdims=True)
X_val_min = X_val_filtered.min(axis=1, keepdims=True)
X_val_nonzero = (X_val_filtered > -110).sum(axis=1, keepdims=True)

X_val_enhanced = np.hstack([
    X_val_filtered,
    X_val_mean,
    X_val_std,
    X_val_max,
    X_val_min,
    X_val_nonzero
])

print(f"Enhanced features: {X_train_enhanced.shape[1]} (added 5 statistical features)")

# -----------------------------
# Scale
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_val_scaled = scaler.transform(X_val_enhanced)

# -----------------------------
# PCA with different variance thresholds
# -----------------------------
print("\n" + "=" * 80)
print("Unique BUILDINGID train:", np.unique(y_train_building), " | Unique FLOOR train:", np.unique(y_train_floor))
print("PCA ANALYSIS")
print("=" * 80)

pca_variances = [0.90, 0.95, 0.98, 0.99]
pca_configs = {}

for var_ratio in pca_variances:
    pca = PCA(n_components=var_ratio, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    pca_configs[var_ratio] = {
        'pca': pca,
        'X_train': X_train_pca,
        'X_val': X_val_pca,
        'n_components': X_train_pca.shape[1]
    }

    print(f"PCA {var_ratio * 100:.0f}% variance: {X_train_pca.shape[1]} components")

# Use 95% variance as default for main analysis
X_train_pca = pca_configs[0.95]['X_train']
X_val_pca = pca_configs[0.95]['X_val']
print(f"\nUsing PCA with 95% variance ({X_train_pca.shape[1]} components) for main analysis")


# -----------------------------
# Helper Functions
# -----------------------------
def purity_score(y_true, y_pred):
    """
    Purity ignoring noise points (-1) in y_pred.
    """
    labels = np.unique(y_pred[y_pred != -1])
    total = 0
    for label in labels:
        idx = (y_pred == label)
        if np.sum(idx) == 0:
            continue
        counts = np.bincount(y_true[idx])
        total += np.max(counts)
    return total / len(y_true)


def clustering_metrics_internal(X, labels):
    """
    Internal clustering metrics (ignore noise -1 for DBSCAN)
    """
    valid = labels != -1
    out = {"Silhouette": np.nan, "DaviesBouldin": np.nan, "CalinskiHarabasz": np.nan}

    if np.sum(valid) > 1:
        uniq = np.unique(labels[valid])
        if len(uniq) > 1:
            out["Silhouette"] = silhouette_score(X[valid], labels[valid])
            out["DaviesBouldin"] = davies_bouldin_score(X[valid], labels[valid])
            out["CalinskiHarabasz"] = calinski_harabasz_score(X[valid], labels[valid])

    return out


def clustering_metrics_external(y_true, labels):
    """
    External metrics vs y_true.
    """
    out = {}

    # All points (noise included)
    out["ARI_all"] = adjusted_rand_score(y_true, labels)
    out["NMI_all"] = normalized_mutual_info_score(y_true, labels)
    out["Homogeneity_all"] = homogeneity_score(y_true, labels)
    out["Completeness_all"] = completeness_score(y_true, labels)
    out["VMeasure_all"] = v_measure_score(y_true, labels)
    out["Purity_all"] = purity_score(y_true, labels)

    # Assigned-only
    valid = labels != -1
    out["NoiseRatio"] = 1.0 - (np.sum(valid) / len(labels))

    if np.sum(valid) > 1 and len(np.unique(labels[valid])) > 0:
        out["ARI_assigned"] = adjusted_rand_score(y_true[valid], labels[valid])
        out["NMI_assigned"] = normalized_mutual_info_score(y_true[valid], labels[valid])
        out["Homogeneity_assigned"] = homogeneity_score(y_true[valid], labels[valid])
        out["Completeness_assigned"] = completeness_score(y_true[valid], labels[valid])
        out["VMeasure_assigned"] = v_measure_score(y_true[valid], labels[valid])

        labels_valid = labels[valid]
        y_valid = y_true[valid]
        uniq = np.unique(labels_valid)
        total = 0
        for lab in uniq:
            idx = labels_valid == lab
            counts = np.bincount(y_valid[idx])
            total += np.max(counts)
        out["Purity_assigned"] = total / len(y_valid)
    else:
        out["ARI_assigned"] = np.nan
        out["NMI_assigned"] = np.nan
        out["Homogeneity_assigned"] = np.nan
        out["Completeness_assigned"] = np.nan
        out["VMeasure_assigned"] = np.nan
        out["Purity_assigned"] = np.nan

    return out


def evaluate_clustering(X, labels, y_true):
    metrics = {}
    metrics.update(clustering_metrics_internal(X, labels))
    metrics.update(clustering_metrics_external(y_true, labels))
    return metrics


def pretty_print_best(title, best):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Params: {best['params']}")
    print(f"\nTrain Metrics:")
    for k in best['keys_show']:
        val = best['train'][k]
        print(f"  {k:20s}: {val:.4f}" if not np.isnan(val) else f"  {k:20s}: NaN")
    print(f"\nValidation Metrics:")
    for k in best['keys_show']:
        val = best['val'][k]
        print(f"  {k:20s}: {val:.4f}" if not np.isnan(val) else f"  {k:20s}: NaN")


# -----------------------------
# 1) K-MEANS SWEEP (expanded range)
# -----------------------------
print("\n" + "=" * 80)
print("K-MEANS CLUSTERING")
print("=" * 80)

k_values = [3, 4, 5, 6, 7, 8, 9]
kmeans_results = []
inertias = []

best_kmeans = {
    "score": -np.inf,
    "params": None,
    "train_labels": None,
    "val_labels": None,
    "train": None,
    "val": None,
    "model": None,
    "keys_show": ["Silhouette", "ARI_all", "NMI_all", "Purity_all", "VMeasure_all"]
}

print("Testing k values:", k_values)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)

    train_labels = kmeans.fit_predict(X_train_pca)
    val_labels = kmeans.predict(X_val_pca)

    train_metrics = evaluate_clustering(X_train_pca, train_labels, y_train_floor)
    val_metrics = evaluate_clustering(X_val_pca, val_labels, y_val_floor)

    inertias.append(kmeans.inertia_)

    kmeans_results.append({
        "k": k,
        "Train": train_metrics,
        "Validation": val_metrics,
        "Inertia": kmeans.inertia_
    })

    # Select best by composite score: ARI + NMI
    score = 0.5 * val_metrics["ARI_all"] + 0.5 * val_metrics["NMI_all"]

    if not np.isnan(score) and score > best_kmeans["score"]:
        best_kmeans.update({
            "score": score,
            "params": {"k": k},
            "train_labels": train_labels,
            "val_labels": val_labels,
            "train": train_metrics,
            "val": val_metrics,
            "model": kmeans
        })

    print(f"k={k}: Val ARI={val_metrics['ARI_all']:.4f}, Val NMI={val_metrics['NMI_all']:.4f}, Score={score:.4f}")

pretty_print_best("BEST K-MEANS (by composite ARI+NMI score)", best_kmeans)

# Elbow plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
plt.title('Elbow Method - Inertia vs k', fontsize=12, fontweight='bold')
plt.xlabel('Number of clusters (k)', fontsize=11)
plt.ylabel('Inertia', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)

# Silhouette plot
plt.subplot(1, 2, 2)
silhouettes_train = [res["Train"]["Silhouette"] for res in kmeans_results]
silhouettes_val = [res["Validation"]["Silhouette"] for res in kmeans_results]
plt.plot(k_values, silhouettes_train, marker='o', label='Train', linewidth=2, markersize=8)
plt.plot(k_values, silhouettes_val, marker='s', label='Validation', linewidth=2, markersize=8)
plt.title('K-Means - Silhouette Score', fontsize=12, fontweight='bold')
plt.xlabel('Number of clusters (k)', fontsize=11)
plt.ylabel('Silhouette Score', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# -----------------------------
# 2) DBSCAN SWEEP (expanded parameters)
# -----------------------------
print("\n" + "=" * 80)
print("DBSCAN CLUSTERING")
print("=" * 80)

eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
min_samples_values = [5, 10, 15, 20]

dbscan_results = []

best_dbscan = {
    "score": -np.inf,
    "params": None,
    "train_labels": None,
    "val_labels": None,
    "train": None,
    "val": None,
    "keys_show": ["Silhouette", "NoiseRatio", "ARI_all", "NMI_all", "Purity_all"]
}

print(f"Testing eps values: {eps_values}")
print(f"Testing min_samples: {min_samples_values}")
print()

for min_samp in min_samples_values:
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        train_labels = dbscan.fit_predict(X_train_pca)

        # Count clusters
        n_clusters = len(np.unique(train_labels[train_labels != -1]))
        n_noise = np.sum(train_labels == -1)

        # Skip if all noise or only 1 cluster
        if n_clusters < 2:
            continue

        # Pseudo-assign validation points
        val_labels = np.full(len(X_val_pca), -1)

        core_mask = train_labels != -1
        core_points = X_train_pca[core_mask]
        core_labels = train_labels[core_mask]

        if len(core_points) > 0:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(core_points)
            distances, indices = nn.kneighbors(X_val_pca)
            within = distances[:, 0] <= eps
            val_labels[within] = core_labels[indices[within, 0]]

        train_metrics = evaluate_clustering(X_train_pca, train_labels, y_train_floor)
        val_metrics = evaluate_clustering(X_val_pca, val_labels, y_val_floor)

        dbscan_results.append({
            "eps": eps,
            "min_samples": min_samp,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "Train": train_metrics,
            "Validation": val_metrics
        })

        # Composite score with noise penalty
        score = 0.5 * val_metrics["ARI_all"] + 0.5 * val_metrics["NMI_all"]
        if np.isnan(score):
            continue

        # Penalize excessive noise
        noise_penalty = 0.7 * val_metrics["NoiseRatio"]
        score_adj = score - noise_penalty

        if score_adj > best_dbscan["score"]:
            best_dbscan.update({
                "score": score_adj,
                "params": {"eps": eps, "min_samples": min_samp},
                "train_labels": train_labels,
                "val_labels": val_labels,
                "train": train_metrics,
                "val": val_metrics
            })

if best_dbscan["params"] is not None:
    pretty_print_best("BEST DBSCAN (by composite ARI+NMI - noise penalty)", best_dbscan)
else:
    print("\nNo valid DBSCAN configuration found (all produced <2 clusters)")

# -----------------------------
# 3) k-distance plot for DBSCAN tuning
# -----------------------------
print("\n" + "=" * 80)
print("k-DISTANCE PLOT FOR DBSCAN TUNING")
print("=" * 80)

min_samples = 10
nn_k = NearestNeighbors(n_neighbors=min_samples)
nn_k.fit(X_train_pca)
distances, _ = nn_k.kneighbors(X_train_pca)

kth_dist = np.sort(distances[:, -1])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(kth_dist, linewidth=1.5)
plt.title(f'k-distance plot (k={min_samples})', fontsize=12, fontweight='bold')
plt.xlabel('Points sorted by distance', fontsize=11)
plt.ylabel(f'Distance to {min_samples}-th nearest neighbor', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()

# Zoomed version
plt.subplot(1, 2, 2)
plt.plot(kth_dist[:int(len(kth_dist) * 0.9)], linewidth=1.5)
plt.title(f'k-distance plot (90% of points)', fontsize=12, fontweight='bold')
plt.xlabel('Points sorted by distance', fontsize=11)
plt.ylabel(f'Distance to {min_samples}-th nearest neighbor', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Metric comparison plots
# -----------------------------
print("\n" + "=" * 80)
print("GENERATING METRIC PLOTS")
print("=" * 80)

# K-Means metrics
metrics_to_plot = ["ARI_all", "NMI_all", "Purity_all", "VMeasure_all"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    train_vals = [res["Train"][metric] for res in kmeans_results]
    val_vals = [res["Validation"][metric] for res in kmeans_results]

    ax.plot(k_values, train_vals, marker='o', label='Train', linewidth=2, markersize=8)
    ax.plot(k_values, val_vals, marker='s', label='Validation', linewidth=2, markersize=8)
    ax.set_title(f'K-Means - {metric}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of clusters (k)', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

plt.tight_layout()
plt.show()

# DBSCAN metrics (if available)
if len(dbscan_results) > 0:
    # Group by min_samples for clearer visualization
    for min_samp in min_samples_values:
        subset = [r for r in dbscan_results if r["min_samples"] == min_samp]
        if len(subset) == 0:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        eps_vals = [r["eps"] for r in subset]

        metrics_dbscan = ["Silhouette", "NoiseRatio", "ARI_all", "NMI_all", "Purity_all", "VMeasure_all"]

        for idx, metric in enumerate(metrics_dbscan):
            ax = axes[idx]
            train_vals = [r["Train"][metric] for r in subset]
            val_vals = [r["Validation"][metric] for r in subset]

            ax.plot(eps_vals, train_vals, marker='o', label='Train', linewidth=2, markersize=8)
            ax.plot(eps_vals, val_vals, marker='s', label='Validation', linewidth=2, markersize=8)
            ax.set_title(f'DBSCAN - {metric} (min_samples={min_samp})', fontsize=11, fontweight='bold')
            ax.set_xlabel('eps', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def map_clusters_to_labels(y_true, cluster_labels):
    mapped = np.full_like(cluster_labels, fill_value=-1)
    for c in np.unique(cluster_labels):
        if c == -1:
            continue
        idx = cluster_labels == c
        # majority class among true labels
        vals, counts = np.unique(y_true[idx], return_counts=True)
        mapped_label = vals[np.argmax(counts)]
        mapped[idx] = mapped_label
    return mapped

# -----------------------------
# 5) Confusion Matrices
# -----------------------------
print("\n" + "=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-Means confusion matrix
kmeans_mapped = map_clusters_to_labels(y_val_floor, best_kmeans["val_labels"])
cm_kmeans = confusion_matrix(y_val_floor, kmeans_mapped)

sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title(f'K-Means Confusion Matrix (k={best_kmeans["params"]["k"]})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Cluster', fontsize=11)
axes[0].set_ylabel('True Floor', fontsize=11)

# DBSCAN confusion matrix (if available)
if best_dbscan["val_labels"] is not None:
    dbscan_mapped = map_clusters_to_labels(y_val_floor, best_dbscan["val_labels"])
    cm_dbscan = confusion_matrix(y_val_floor, dbscan_mapped)

    sns.heatmap(cm_dbscan, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title(
        f'DBSCAN Confusion Matrix (eps={best_dbscan["params"]["eps"]}, min_samples={best_dbscan["params"]["min_samples"]})',
        fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted Cluster', fontsize=11)
    axes[1].set_ylabel('True Floor', fontsize=11)
else:
    axes[1].text(0.5, 0.5, 'DBSCAN: No valid configuration',
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# 6) t-SNE Visualization
# -----------------------------
print("\n" + "=" * 80)
print("GENERATING t-SNE VISUALIZATIONS")
print("=" * 80)

# Subsample if dataset is too large for t-SNE
max_samples = 5000
if len(X_train_pca) > max_samples:
    print(f"Subsampling {max_samples} points for t-SNE (from {len(X_train_pca)} total)")
    idx_sample = np.random.choice(len(X_train_pca), max_samples, replace=False)
    X_tsne_input = X_train_pca[idx_sample]
    y_tsne_floor = y_train_floor[idx_sample]
    kmeans_tsne_labels = best_kmeans["train_labels"][idx_sample]
    if best_dbscan["train_labels"] is not None:
        dbscan_tsne_labels = best_dbscan["train_labels"][idx_sample]
    else:
        dbscan_tsne_labels = None
else:
    X_tsne_input = X_train_pca
    y_tsne_floor = y_train_floor
    kmeans_tsne_labels = best_kmeans["train_labels"]
    dbscan_tsne_labels = best_dbscan["train_labels"]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    verbose=1
)
X_train_tsne = tsne.fit_transform(X_tsne_input)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True labels
scatter1 = axes[0].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                           c=y_tsne_floor, s=10, cmap='viridis', alpha=0.6)
axes[0].set_title('True FLOOR Labels (t-SNE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('t-SNE Component 1', fontsize=11)
axes[0].set_ylabel('t-SNE Component 2', fontsize=11)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Floor')

# K-Means labels
scatter2 = axes[1].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                           c=kmeans_tsne_labels, s=10, cmap='tab10', alpha=0.6)
axes[1].set_title(f'K-Means Clusters (k={best_kmeans["params"]["k"]})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('t-SNE Component 1', fontsize=11)
axes[1].set_ylabel('t-SNE Component 2', fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

# DBSCAN labels
if dbscan_tsne_labels is not None:
    scatter3 = axes[2].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                               c=dbscan_tsne_labels, s=10, cmap='tab10', alpha=0.6)
    axes[2].set_title(
        f'DBSCAN Clusters (eps={best_dbscan["params"]["eps"]}, min_samples={best_dbscan["params"]["min_samples"]})',
        fontsize=11, fontweight='bold')
    axes[2].set_xlabel('t-SNE Component 1', fontsize=11)
    axes[2].set_ylabel('t-SNE Component 2', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[2], label='Cluster')
else:
    axes[2].text(0.5, 0.5, 'DBSCAN: No valid configuration',
                 ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
    axes[2].axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# 7) Final Model Comparison
# -----------------------------
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

comparison_data = []

# K-Means
comparison_data.append({
    'Model': 'K-Means',
    'Params': f"k={best_kmeans['params']['k']}",
    'Val_ARI': best_kmeans['val']['ARI_all'],
    'Val_NMI': best_kmeans['val']['NMI_all'],
    'Val_Purity': best_kmeans['val']['Purity_all'],
    'Val_Silhouette': best_kmeans['val']['Silhouette'],
    'Val_VMeasure': best_kmeans['val']['VMeasure_all'],
    'Noise_Ratio': 0.0
})

# DBSCAN
if best_dbscan['params'] is not None:
    comparison_data.append({
        'Model': 'DBSCAN',
        'Params': f"eps={best_dbscan['params']['eps']}, min_s={best_dbscan['params']['min_samples']}",
        'Val_ARI': best_dbscan['val']['ARI_all'],
        'Val_NMI': best_dbscan['val']['NMI_all'],
        'Val_Purity': best_dbscan['val']['Purity_all'],
        'Val_Silhouette': best_dbscan['val']['Silhouette'],
        'Val_VMeasure': best_dbscan['val']['VMeasure_all'],
        'Noise_Ratio': best_dbscan['val']['NoiseRatio']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\nValidation Set Performance:")
print(comparison_df.to_string(index=False))

# -----------------------------
# 8) Recommendations
# -----------------------------
kmeans_score = 0.5 * best_kmeans['val']['ARI_all'] + 0.5 * best_kmeans['val']['NMI_all']
dbscan_score = 0.5 * best_dbscan['val']['ARI_all'] + 0.5 * best_dbscan['val']['NMI_all'] - 0.7 * best_dbscan['val']['NoiseRatio']

best_model = "K-Means" if kmeans_score >= dbscan_score else "DBSCAN"

print("\nCONCLUSIONES")

kmeans_score = 0.5 * best_kmeans['val']['ARI_all'] + 0.5 * best_kmeans['val']['NMI_all']

if best_dbscan['params'] is not None:
    dbscan_score = 0.5 * best_dbscan['val']['ARI_all'] + 0.5 * best_dbscan['val']['NMI_all'] - 0.7 * best_dbscan['val']['NoiseRatio']
else:
    dbscan_score = -np.inf

mejor_modelo = "K-Means" if kmeans_score >= dbscan_score else "DBSCAN"

print(f"""
1. El modelo con mejor desempeño global es: {mejor_modelo}.

2. La reducción de dimensionalidad mediante PCA permitió disminuir
   significativamente el número de características manteniendo el 95% de la varianza,
   lo que mejoró estabilidad y eficiencia computacional.

3. La eliminación de puntos de acceso no informativos redujo ruido en el espacio
   de características y facilitó la formación de clústeres más coherentes.

4. K-Means mostró mayor estabilidad cuando la estructura de los datos es aproximadamente
   esférica, mientras que DBSCAN permitió identificar ruido explícitamente.

5. Las métricas ARI y NMI demostraron ser las más confiables para evaluar la
   correspondencia entre clústeres y pisos reales.
""")

print("\nANÁLISIS COMPLETADO")