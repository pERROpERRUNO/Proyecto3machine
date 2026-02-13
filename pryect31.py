import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score,)
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 'BUILDINGID' | 'FLOOR' | 'AREAID'
K_SELECTION_TARGET = 'BUILDINGID'

# FASE 1: CARGA Y PREPROCESAMIENTO DE DATOS
print("FASE 1: CARGA Y PREPROCESAMIENTO DE DATOS")
print()

train_file = "trainingData.csv"
val_file = "validationData.csv"

print("[1.1] Cargando datasets...")
try:
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Validation data: {val_data.shape}")
except FileNotFoundError as e:
    print(f"ERROR: No se encontró el archivo")
    exit(1)

print()

# Identificar columnas
wap_columns = [col for col in train_data.columns if col.startswith('WAP')]
metadata_columns = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID', 'SPACEID',
                    'RELATIVEPOSITION', 'USERID', 'PHONEID', 'TIMESTAMP']

train_data['AREAID'] = (
    train_data['BUILDINGID'].astype(str) + '-' +
    train_data['FLOOR'].astype(str) + '-' +
    train_data['SPACEID'].astype(str)
)
val_data['AREAID'] = (
    val_data['BUILDINGID'].astype(str) + '-' +
    val_data['FLOOR'].astype(str) + '-' +
    val_data['SPACEID'].astype(str)
)

print("[1.2] Estructura del dataset:")
print(f"- WAPs detectados: {len(wap_columns)}")
print(f"- Edificios únicos: {train_data['BUILDINGID'].nunique()}")
print(f"- Pisos únicos: {train_data['FLOOR'].nunique()}")
print(f"- Áreas únicas (B-F-SPACE): {train_data['AREAID'].nunique()}")
print()

# Preprocesamiento
print("[1.3] Preprocesamiento...")
X_train = train_data[wap_columns].copy()
X_val = val_data[wap_columns].copy()

# RSSI: 100 = no detectado -> lo llevamos a -110 (valor típico de señal muy débil)
X_train = X_train.replace(100, -110)
X_val = X_val.replace(100, -110)

print(f"✓ Rango señal: [{X_train.values.min():.0f}, {X_train.values.max():.0f}]")
print(f"✓ Media: {X_train.values.mean():.2f}, Std: {X_train.values.std():.2f}")
print()

# Filtrar WAPs por varianza
variance_threshold = 1.0
wap_variances = X_train.var()
active_waps = wap_variances[wap_variances > variance_threshold].index.tolist()
print(f"[1.4] WAPs activos: {len(active_waps)} de {len(wap_columns)}")

X_train = X_train[active_waps]
X_val = X_val[active_waps]

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print(f"✓ Shape final: {X_train_scaled.shape}")
print()

# FASE 2: ANÁLISIS EXPLORATORIO
print("FASE 2: ANÁLISIS EXPLORATORIO")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

building_counts = train_data['BUILDINGID'].value_counts().sort_index()
axes[0].bar(building_counts.index, building_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Edificio ID', fontsize=12)
axes[0].set_ylabel('Número de Muestras', fontsize=12)
axes[0].set_title('Distribución por Edificio', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

floor_counts = train_data['FLOOR'].value_counts().sort_index()
axes[1].bar(floor_counts.index, floor_counts.values, color='coral', alpha=0.7)
axes[1].set_xlabel('Piso ID', fontsize=12)
axes[1].set_ylabel('Número de Muestras', fontsize=12)
axes[1].set_title('Distribución por Piso', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('01_analisis_exploratorio.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 01_analisis_exploratorio.png")
print()

# FASE 3: REDUCCIÓN DE DIMENSIONALIDAD
print("FASE 3: REDUCCIÓN DE DIMENSIONALIDAD")
print()

print("[3.1] PCA...")
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_train_scaled)

cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
print(f"✓ Componentes para 95% varianza: {n_components_95}")

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train_scaled)
print(f"✓ Varianza 2D: {pca_2d.explained_variance_ratio_.sum() * 100:.2f}%")

pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_train_scaled)
print()

print("[3.2] t-SNE (puede tardar)...")
sample_size = min(5000, len(X_train_scaled))
sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train_scaled[sample_indices])
print("✓ t-SNE completado")
print()

# Visualización
fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=train_data['BUILDINGID'], cmap='viridis', alpha=0.6, s=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title('PCA 2D - Edificios')
plt.colorbar(scatter1, ax=ax1, label='Building ID')

ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=train_data['FLOOR'], cmap='plasma', alpha=0.6, s=10)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
ax2.set_title('PCA 2D - Pisos')
plt.colorbar(scatter2, ax=ax2, label='Floor ID')

ax3 = plt.subplot(2, 3, 3)
ax3.plot(range(1, min(200, len(cumsum_variance)) + 1), cumsum_variance[:200], linewidth=2)
ax3.axhline(y=0.95, color='r', linestyle='--', label='95%')
ax3.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} comp')
ax3.set_xlabel('Componentes'); ax3.set_ylabel('Varianza Acum.')
ax3.set_title('PCA: Varianza Explicada')
ax3.legend(); ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
scatter4 = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1],
                       c=train_data.iloc[sample_indices]['BUILDINGID'],
                       cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('t-SNE 1'); ax4.set_ylabel('t-SNE 2')
ax4.set_title('t-SNE - Edificios')
plt.colorbar(scatter4, ax=ax4, label='Building ID')

ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(X_tsne[:, 0], X_tsne[:, 1],
                       c=train_data.iloc[sample_indices]['FLOOR'],
                       cmap='plasma', alpha=0.6, s=10)
ax5.set_xlabel('t-SNE 1'); ax5.set_ylabel('t-SNE 2')
ax5.set_title('t-SNE - Pisos')
plt.colorbar(scatter5, ax=ax5, label='Floor ID')

ax6 = plt.subplot(2, 3, 6, projection='3d')
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_train_scaled)
scatter6 = ax6.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                       c=train_data['BUILDINGID'], cmap='viridis', alpha=0.5, s=5)
ax6.set_xlabel('PC1'); ax6.set_ylabel('PC2'); ax6.set_zlabel('PC3')
ax6.set_title('PCA 3D - Edificios')

plt.tight_layout()
plt.savefig('02_reduccion_dimensionalidad.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 02_reduccion_dimensionalidad.png")
print()

# FASE 4: K-MEANS
print("FASE 4: K-MEANS CON SELECCIÓN ÓPTIMA DE K")
print()
print("[4.1] Evaluando K desde 2 hasta 20...")
k_range = range(2, 21)
k_candidates = [2, 3, 5, 8, 15, 20]

inertias = []
silhouette_scores_list = []
davies_bouldin_scores_list = []
calinski_harabasz_scores_list = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    inertias.append(kmeans.inertia_)
    silhouette_scores_list.append(silhouette_score(X_pca, labels))
    davies_bouldin_scores_list.append(davies_bouldin_score(X_pca, labels))
    calinski_harabasz_scores_list.append(calinski_harabasz_score(X_pca, labels))

print("✓ Métricas internas calculadas")
print()

print("[4.2] Evaluando K candidatos vs ground truth (BUILDING / FLOOR / AREA)...")
ari_building_scores, nmi_building_scores, vmeasure_building_scores = [], [], []
ari_floor_scores, nmi_floor_scores, vmeasure_floor_scores = [], [], []
ari_area_scores, nmi_area_scores, vmeasure_area_scores = [], [], []

for k in k_candidates:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_pca)

    # BUILDING
    ari_building_scores.append(adjusted_rand_score(train_data['BUILDINGID'], labels_temp))
    nmi_building_scores.append(normalized_mutual_info_score(train_data['BUILDINGID'], labels_temp))
    vmeasure_building_scores.append(v_measure_score(train_data['BUILDINGID'], labels_temp))

    # FLOOR
    ari_floor_scores.append(adjusted_rand_score(train_data['FLOOR'], labels_temp))
    nmi_floor_scores.append(normalized_mutual_info_score(train_data['FLOOR'], labels_temp))
    vmeasure_floor_scores.append(v_measure_score(train_data['FLOOR'], labels_temp))

    # AREAID = BUILDING-FLOOR-SPACE
    ari_area_scores.append(adjusted_rand_score(train_data['AREAID'], labels_temp))
    nmi_area_scores.append(normalized_mutual_info_score(train_data['AREAID'], labels_temp))
    vmeasure_area_scores.append(v_measure_score(train_data['AREAID'], labels_temp))

# Seleccionar K óptimo
if K_SELECTION_TARGET == 'FLOOR':
    optimal_k = k_candidates[int(np.argmax(nmi_floor_scores))]
    reason = f"Máximo NMI con FLOOR = {max(nmi_floor_scores):.4f}"
elif K_SELECTION_TARGET == 'AREAID':
    optimal_k = k_candidates[int(np.argmax(nmi_area_scores))]
    reason = f"Máximo NMI con AREAID = {max(nmi_area_scores):.4f}"
else:
    optimal_k = k_candidates[int(np.argmax(nmi_building_scores))]
    reason = f"Máximo NMI con BUILDINGID = {max(nmi_building_scores):.4f}"

print(f"✓ K ÓPTIMO: {optimal_k}")
print(f"  Razón: {reason}")
print()

# Gráfico selección K
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(list(k_range), inertias, 'o-', linewidth=2)
axes[0, 0].set_xlabel('K'); axes[0, 0].set_ylabel('Inercia')
axes[0, 0].set_title('Método del Codo')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(list(k_range), silhouette_scores_list, 'o-', linewidth=2, color='green')
axes[0, 1].set_xlabel('K'); axes[0, 1].set_ylabel('Silhouette')
axes[0, 1].set_title('Silhouette Score')
axes[0, 1].grid(alpha=0.3)

axes[0, 2].plot(list(k_range), davies_bouldin_scores_list, 'o-', linewidth=2, color='red')
axes[0, 2].set_xlabel('K'); axes[0, 2].set_ylabel('Davies-Bouldin')
axes[0, 2].set_title('Davies-Bouldin (menor=mejor)')
axes[0, 2].grid(alpha=0.3)

axes[1, 0].plot(list(k_range), calinski_harabasz_scores_list, 'o-', linewidth=2, color='purple')
axes[1, 0].set_xlabel('K'); axes[1, 0].set_ylabel('Calinski-Harabasz')
axes[1, 0].set_title('Calinski-Harabasz')
axes[1, 0].grid(alpha=0.3)

# Externa (BUILDING) para candidatos
axes[1, 1].plot(k_candidates, nmi_building_scores, 's-', linewidth=2, color='green', label='NMI(Build)', markersize=8)
axes[1, 1].plot(k_candidates, ari_building_scores, 'o-', linewidth=2, color='blue', label='ARI(Build)', markersize=8)
n_buildings = train_data['BUILDINGID'].nunique()
axes[1, 1].axvline(x=n_buildings, color='red', linestyle='--', alpha=0.5, label=f'K={n_buildings} (edificios)')
axes[1, 1].axvline(x=optimal_k, color='purple', linewidth=2, label=f'Óptimo={optimal_k}')
axes[1, 1].set_xlabel('K'); axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Alineación con Building ID (candidatos)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

axes[1, 2].axis('off')
idx_ok = k_candidates.index(optimal_k)
text_summary = f"""
K ÓPTIMO SELECCIONADO: {optimal_k}
Selección basada en: {K_SELECTION_TARGET}

Métricas para K={optimal_k}:
BUILDING:
• ARI: {ari_building_scores[idx_ok]:.4f}
• NMI: {nmi_building_scores[idx_ok]:.4f}
• V:   {vmeasure_building_scores[idx_ok]:.4f}

FLOOR:
• ARI: {ari_floor_scores[idx_ok]:.4f}
• NMI: {nmi_floor_scores[idx_ok]:.4f}
• V:   {vmeasure_floor_scores[idx_ok]:.4f}

AREA (B-F-SPACE):
• ARI: {ari_area_scores[idx_ok]:.4f}
• NMI: {nmi_area_scores[idx_ok]:.4f}
• V:   {vmeasure_area_scores[idx_ok]:.4f}

Ground Truth:
• Edificios: {n_buildings}
• Pisos: {train_data['FLOOR'].nunique()}
• Áreas: {train_data['AREAID'].nunique()}
"""
axes[1, 2].text(0.02, 0.5, text_summary, fontsize=10, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig('03_kmeans_seleccion_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 03_kmeans_seleccion_k.png")
print()

# Aplicar K-Means final
print(f"[4.3] Aplicando K-Means con K={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_pca)
print(f"✓ Distribución: {np.bincount(kmeans_labels)}")
print()

# Visualización resultados K-Means
fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=kmeans_labels, cmap='tab10', alpha=0.6, s=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title(f'K-Means (K={optimal_k})')
plt.colorbar(scatter1, ax=ax1, label='Cluster')

ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1],
                       c=kmeans_labels[sample_indices], cmap='tab10', alpha=0.6, s=10)
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
ax2.set_title('K-Means en t-SNE')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

ax3 = plt.subplot(2, 3, 3)
cluster_sizes = np.bincount(kmeans_labels)
ax3.bar(range(len(cluster_sizes)), cluster_sizes, color='steelblue', alpha=0.7)
ax3.set_xlabel('Cluster ID'); ax3.set_ylabel('Tamaño')
ax3.set_title('Distribución de Clusters')
ax3.grid(axis='y', alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
scatter4 = ax4.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=train_data['BUILDINGID'], cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('PC1'); ax4.set_ylabel('PC2')
ax4.set_title('Ground Truth: Edificios')
plt.colorbar(scatter4, ax=ax4, label='Building ID')

ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=train_data['FLOOR'], cmap='plasma', alpha=0.6, s=10)
ax5.set_xlabel('PC1'); ax5.set_ylabel('PC2')
ax5.set_title('Ground Truth: Pisos')
plt.colorbar(scatter5, ax=ax5, label='Floor ID')

ax6 = plt.subplot(2, 3, 6)
confusion_matrix_build = pd.crosstab(train_data['BUILDINGID'], kmeans_labels)
sns.heatmap(confusion_matrix_build, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
ax6.set_xlabel('Cluster K-Means'); ax6.set_ylabel('Building ID Real')
ax6.set_title('Matriz de Confusión (Building vs Cluster)')

plt.tight_layout()
plt.savefig('04_kmeans_resultados.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 04_kmeans_resultados.png")
print()

# FASE 5: DBSCAN

print("FASE 5: DBSCAN CON OPTIMIZACIÓN DE PARÁMETROS")
print()
print("[5.1] Estimando eps con K-distance...")
k = 10
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_pca)
distances, _ = neighbors.kneighbors(X_pca)

k_distances = np.sort(distances[:, -1])
eps_estimated = k_distances[int(len(k_distances) * 0.95)]
print(f"✓ Eps estimado (percentil 95): {eps_estimated:.4f}")
print()

print("[5.2] Probando configuraciones DBSCAN...")
eps_values = [eps_estimated / 3, eps_estimated / 2, eps_estimated, eps_estimated * 1.5]
min_samples_values = [3, 5, 10, 15]

dbscan_results = []
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_ratio = n_noise / len(labels)
        coverage = 1.0 - noise_ratio

        sil = 0.0
        nmi_build = vmeas_build = 0.0
        nmi_floor = vmeas_floor = 0.0
        nmi_area = vmeas_area = 0.0

        if n_clusters > 1:
            mask = labels != -1
            # Silhouette requiere >1 clusters y >1 puntos
            if mask.sum() > 2 and len(np.unique(labels[mask])) > 1:
                sil = float(silhouette_score(X_pca[mask], labels[mask]))

            # Externa (sin ruido)
            if mask.sum() > 0:
                nmi_build = float(normalized_mutual_info_score(train_data.loc[mask, 'BUILDINGID'], labels[mask]))
                vmeas_build = float(v_measure_score(train_data.loc[mask, 'BUILDINGID'], labels[mask]))

                nmi_floor = float(normalized_mutual_info_score(train_data.loc[mask, 'FLOOR'], labels[mask]))
                vmeas_floor = float(v_measure_score(train_data.loc[mask, 'FLOOR'], labels[mask]))

                nmi_area = float(normalized_mutual_info_score(train_data.loc[mask, 'AREAID'], labels[mask]))
                vmeas_area = float(v_measure_score(train_data.loc[mask, 'AREAID'], labels[mask]))

        score = (0.55 * nmi_area) + (0.30 * coverage) + (0.15 * max(sil, 0.0))
        dbscan_results.append({
            'eps': float(eps),
            'min_samples': int(min_samples),
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'noise_ratio': float(noise_ratio),
            'coverage': float(coverage),
            'silhouette': float(sil),
            'nmi_building': float(nmi_build),
            'vmeasure_building': float(vmeas_build),
            'nmi_floor': float(nmi_floor),
            'vmeasure_floor': float(vmeas_floor),
            'nmi_area': float(nmi_area),
            'vmeasure_area': float(vmeas_area),
            'score': float(score)
        })

dbscan_df = pd.DataFrame(dbscan_results)
print("✓ Configuraciones evaluadas")
print()

best_idx = dbscan_df['score'].idxmax()
best_config = dbscan_df.loc[best_idx]
optimal_eps = float(best_config['eps'])
optimal_min_samples = int(best_config['min_samples'])

print("✓ PARÁMETROS ÓPTIMOS:")
print(f"  eps: {optimal_eps:.4f}")
print(f"  min_samples: {optimal_min_samples}")
print(f"  Clusters esperados: {int(best_config['n_clusters'])}")
print(f"  Ruido esperado: {best_config['noise_ratio'] * 100:.1f}%")
print(f"  Cobertura: {best_config['coverage'] * 100:.1f}%")
print(f"  NMI (AreaID): {best_config['nmi_area']:.4f}")
print(f"  NMI (Building): {best_config['nmi_building']:.4f}")
print()

# Gráfico selección parámetros DBSCAN
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(range(len(k_distances)), k_distances, linewidth=2)
axes[0, 0].axhline(y=eps_estimated, color='r', linestyle='--', label=f'Eps={eps_estimated:.2f}')
axes[0, 0].set_xlabel('Puntos ordenados')
axes[0, 0].set_ylabel('Distancia k-vecino')
axes[0, 0].set_title('K-Distance Plot')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

for eps in eps_values:
    subset = dbscan_df[dbscan_df['eps'] == eps]
    axes[0, 1].plot(subset['min_samples'], subset['n_clusters'],
                    'o-', linewidth=2, label=f'eps={eps:.1f}')
axes[0, 1].set_xlabel('min_samples')
axes[0, 1].set_ylabel('Número de Clusters')
axes[0, 1].set_title('Clusters vs Parámetros')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

for eps in eps_values:
    subset = dbscan_df[dbscan_df['eps'] == eps]
    axes[1, 0].plot(subset['min_samples'], subset['noise_ratio'] * 100,
                    'o-', linewidth=2, label=f'eps={eps:.1f}')
axes[1, 0].set_xlabel('min_samples')
axes[1, 0].set_ylabel('Ruido (%)')
axes[1, 0].set_title('Ruido vs Parámetros')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

for eps in eps_values:
    subset = dbscan_df[dbscan_df['eps'] == eps]
    axes[1, 1].plot(subset['min_samples'], subset['nmi_area'],
                    'o-', linewidth=2, label=f'eps={eps:.1f}')
axes[1, 1].set_xlabel('min_samples')
axes[1, 1].set_ylabel('NMI con AreaID')
axes[1, 1].set_title('NMI(AreaID) vs Parámetros')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('05_dbscan_seleccion_parametros.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 05_dbscan_seleccion_parametros.png")
print()

# Aplicar DBSCAN final
print("[5.3] Aplicando DBSCAN...")
dbscan_final = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
dbscan_labels = dbscan_final.fit_predict(X_pca)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = int(np.sum(dbscan_labels == -1))

print(f"✓ Clusters: {n_clusters_dbscan}")
print(f"✓ Ruido: {n_noise_dbscan} ({n_noise_dbscan / len(dbscan_labels) * 100:.1f}%)")
print()

# Visualización DBSCAN
fig = plt.figure(figsize=(18, 10))

ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                       c=dbscan_labels, cmap='tab20', alpha=0.6, s=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.set_title(f'DBSCAN (eps={optimal_eps:.1f}, min_samp={optimal_min_samples})')
plt.colorbar(scatter1, ax=ax1, label='Cluster')

ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1],
                       c=dbscan_labels[sample_indices], cmap='tab20', alpha=0.6, s=10)
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
ax2.set_title('DBSCAN en t-SNE')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

ax4 = plt.subplot(2, 3, 4)
hexbin = ax4.hexbin(X_pca_2d[:, 0], X_pca_2d[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
ax4.set_xlabel('PC1'); ax4.set_ylabel('PC2')
ax4.set_title('Densidad de Puntos')
plt.colorbar(hexbin, ax=ax4, label='Densidad')

ax5 = plt.subplot(2, 3, 5)
ax5.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=kmeans_labels, cmap='tab10', alpha=0.35, s=12, marker='o', label='K-Means')
ax5.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=dbscan_labels, cmap='tab20', alpha=0.35, s=12, marker='^', label='DBSCAN')
ax5.set_xlabel('PC1'); ax5.set_ylabel('PC2')
ax5.set_title('Superposición (PCA 2D)')
ax5.legend()


plt.tight_layout()
plt.savefig('06_dbscan_resultados.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 06_dbscan_resultados.png")
print()


# FASE 6: EVALUACIÓN Y COMPARACIÓN
print("FASE 6: EVALUACIÓN Y COMPARACIÓN")
print()

#  K-Means: internas 
sil_kmeans = float(silhouette_score(X_pca, kmeans_labels))
db_kmeans = float(davies_bouldin_score(X_pca, kmeans_labels))
ch_kmeans = float(calinski_harabasz_score(X_pca, kmeans_labels))

# K-Means: externas (BUILDING/FLOOR/AREA)
ari_kmeans_b = float(adjusted_rand_score(train_data['BUILDINGID'], kmeans_labels))
nmi_kmeans_b = float(normalized_mutual_info_score(train_data['BUILDINGID'], kmeans_labels))
vmeas_kmeans_b = float(v_measure_score(train_data['BUILDINGID'], kmeans_labels))

ari_kmeans_f = float(adjusted_rand_score(train_data['FLOOR'], kmeans_labels))
nmi_kmeans_f = float(normalized_mutual_info_score(train_data['FLOOR'], kmeans_labels))
vmeas_kmeans_f = float(v_measure_score(train_data['FLOOR'], kmeans_labels))

ari_kmeans_a = float(adjusted_rand_score(train_data['AREAID'], kmeans_labels))
nmi_kmeans_a = float(normalized_mutual_info_score(train_data['AREAID'], kmeans_labels))
vmeas_kmeans_a = float(v_measure_score(train_data['AREAID'], kmeans_labels))

print(" K-MEANS:")
print(f"  Silhouette: {sil_kmeans:.4f}")
print(f"  Davies-Bouldin: {db_kmeans:.4f}")
print(f"  Calinski-Harabasz: {ch_kmeans:.2f}")
print(f"  ARI/NMI/V (Building): {ari_kmeans_b:.4f} / {nmi_kmeans_b:.4f} / {vmeas_kmeans_b:.4f}")
print(f"  ARI/NMI/V (Floor):    {ari_kmeans_f:.4f} / {nmi_kmeans_f:.4f} / {vmeas_kmeans_f:.4f}")
print(f"  ARI/NMI/V (AreaID):   {ari_kmeans_a:.4f} / {nmi_kmeans_a:.4f} / {vmeas_kmeans_a:.4f}")
print()

# DBSCAN: internas + externas (sin ruido)
mask_dbscan = dbscan_labels != -1
sil_dbscan = db_dbscan = ch_dbscan = 0.0
ari_dbscan_b = nmi_dbscan_b = vmeas_dbscan_b = 0.0
ari_dbscan_f = nmi_dbscan_f = vmeas_dbscan_f = 0.0
ari_dbscan_a = nmi_dbscan_a = vmeas_dbscan_a = 0.0

if mask_dbscan.sum() > 2 and n_clusters_dbscan > 1 and len(np.unique(dbscan_labels[mask_dbscan])) > 1:
    sil_dbscan = float(silhouette_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan]))
    db_dbscan = float(davies_bouldin_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan]))
    ch_dbscan = float(calinski_harabasz_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan]))

    ari_dbscan_b = float(adjusted_rand_score(train_data.loc[mask_dbscan, 'BUILDINGID'], dbscan_labels[mask_dbscan]))
    nmi_dbscan_b = float(normalized_mutual_info_score(train_data.loc[mask_dbscan, 'BUILDINGID'], dbscan_labels[mask_dbscan]))
    vmeas_dbscan_b = float(v_measure_score(train_data.loc[mask_dbscan, 'BUILDINGID'], dbscan_labels[mask_dbscan]))

    ari_dbscan_f = float(adjusted_rand_score(train_data.loc[mask_dbscan, 'FLOOR'], dbscan_labels[mask_dbscan]))
    nmi_dbscan_f = float(normalized_mutual_info_score(train_data.loc[mask_dbscan, 'FLOOR'], dbscan_labels[mask_dbscan]))
    vmeas_dbscan_f = float(v_measure_score(train_data.loc[mask_dbscan, 'FLOOR'], dbscan_labels[mask_dbscan]))

    ari_dbscan_a = float(adjusted_rand_score(train_data.loc[mask_dbscan, 'AREAID'], dbscan_labels[mask_dbscan]))
    nmi_dbscan_a = float(normalized_mutual_info_score(train_data.loc[mask_dbscan, 'AREAID'], dbscan_labels[mask_dbscan]))
    vmeas_dbscan_a = float(v_measure_score(train_data.loc[mask_dbscan, 'AREAID'], dbscan_labels[mask_dbscan]))

    print(" DBSCAN (sin ruido):")
    print(f"  Silhouette: {sil_dbscan:.4f}")
    print(f"  Davies-Bouldin: {db_dbscan:.4f}")
    print(f"  Calinski-Harabasz: {ch_dbscan:.2f}")
    print(f"  ARI/NMI/V (Building): {ari_dbscan_b:.4f} / {nmi_dbscan_b:.4f} / {vmeas_dbscan_b:.4f}")
    print(f"  ARI/NMI/V (Floor):    {ari_dbscan_f:.4f} / {nmi_dbscan_f:.4f} / {vmeas_dbscan_f:.4f}")
    print(f"  ARI/NMI/V (AreaID):   {ari_dbscan_a:.4f} / {nmi_dbscan_a:.4f} / {vmeas_dbscan_a:.4f}")
    print(f"  Cobertura: {mask_dbscan.sum()}/{len(dbscan_labels)} ({mask_dbscan.sum() / len(dbscan_labels) * 100:.1f}%)")
else:
    print(" DBSCAN: N/A (clusters insuficientes o demasiados pocos puntos sin ruido)")

print()

# Gráfico comparativo (interno + externo BUILDING/FLOOR/AREA)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Internas
metrics_internal = ['Silhouette', '1/DB', 'CH/1000']
kmeans_internal = [sil_kmeans, 1 / db_kmeans if db_kmeans > 0 else 0, ch_kmeans / 1000]
dbscan_internal = [sil_dbscan, 1 / db_dbscan if db_dbscan > 0 else 0, ch_dbscan / 1000]

x = np.arange(len(metrics_internal))
width = 0.35
axes[0, 0].bar(x - width / 2, kmeans_internal, width, label='K-Means', alpha=0.8)
axes[0, 0].bar(x + width / 2, dbscan_internal, width, label='DBSCAN', alpha=0.8)
axes[0, 0].set_title('Evaluación Interna')
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(metrics_internal)
axes[0, 0].legend(); axes[0, 0].grid(axis='y', alpha=0.3)

# Externas BUILDING
metrics_external = ['ARI', 'NMI', 'V']
kmeans_build = [ari_kmeans_b, nmi_kmeans_b, vmeas_kmeans_b]
dbscan_build = [ari_dbscan_b, nmi_dbscan_b, vmeas_dbscan_b]

axes[0, 1].bar(x - width / 2, kmeans_build, width, label='K-Means', alpha=0.8)
axes[0, 1].bar(x + width / 2, dbscan_build, width, label='DBSCAN', alpha=0.8)
axes[0, 1].set_title('Externa (BUILDING)')
axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(metrics_external)
axes[0, 1].legend(); axes[0, 1].grid(axis='y', alpha=0.3)

# Externas FLOOR
kmeans_floor = [ari_kmeans_f, nmi_kmeans_f, vmeas_kmeans_f]
dbscan_floor = [ari_dbscan_f, nmi_dbscan_f, vmeas_dbscan_f]

axes[0, 2].bar(x - width / 2, kmeans_floor, width, label='K-Means', alpha=0.8)
axes[0, 2].bar(x + width / 2, dbscan_floor, width, label='DBSCAN', alpha=0.8)
axes[0, 2].set_title('Externa (FLOOR)')
axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(metrics_external)
axes[0, 2].legend(); axes[0, 2].grid(axis='y', alpha=0.3)

# Externas AREAID
kmeans_area = [ari_kmeans_a, nmi_kmeans_a, vmeas_kmeans_a]
dbscan_area = [ari_dbscan_a, nmi_dbscan_a, vmeas_dbscan_a]

axes[1, 0].bar(x - width / 2, kmeans_area, width, label='K-Means', alpha=0.8)
axes[1, 0].bar(x + width / 2, dbscan_area, width, label='DBSCAN', alpha=0.8)
axes[1, 0].set_title('Externa (AREAID = B-F-SPACE)')
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(metrics_external)
axes[1, 0].legend(); axes[1, 0].grid(axis='y', alpha=0.3)

# Distribución (KMeans y DBSCAN)
axes[1, 1].bar(range(optimal_k), np.bincount(kmeans_labels), alpha=0.7, label='K-Means')
axes[1, 1].set_title(f'K-Means: tamaños (K={optimal_k})')
axes[1, 1].set_xlabel('Cluster'); axes[1, 1].set_ylabel('Tamaño')
axes[1, 1].grid(axis='y', alpha=0.3)

# Resumen
axes[1, 2].axis('off')
winner = "K-Means" if nmi_kmeans_a >= nmi_dbscan_a else "DBSCAN"
summary_text = f"""
RESUMEN

K-Means (K={optimal_k}):
• Sil: {sil_kmeans:.4f}
• NMI(Build): {nmi_kmeans_b:.4f}
• NMI(Area):  {nmi_kmeans_a:.4f}

DBSCAN:
• Sil: {sil_dbscan:.4f}
• NMI(Build): {nmi_dbscan_b:.4f}
• NMI(Area):  {nmi_dbscan_a:.4f}
• Ruido: {n_noise_dbscan} ({n_noise_dbscan/len(dbscan_labels)*100:.1f}%)

RECOMENDADO (por NMI AreaID):
{winner}
"""
axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig('07_comparacion_algoritmos.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: 07_comparacion_algoritmos.png")
print()

# Guardar resultados
print("[6.1] Guardando resumen...")
with open('resultados_final.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PROYECTO: CLUSTERING PARA POSICIONAMIENTO INDOOR WiFi\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(f"Muestras: {len(train_data)}\n")
    f.write(f"WAPs activos: {len(active_waps)}\n")
    f.write(f"Componentes PCA (95%): {n_components_95}\n")
    f.write(f"Edificios: {train_data['BUILDINGID'].nunique()}\n")
    f.write(f"Pisos: {train_data['FLOOR'].nunique()}\n")
    f.write(f"Áreas (B-F-SPACE): {train_data['AREAID'].nunique()}\n\n")

    f.write("K-MEANS\n")
    f.write("-" * 80 + "\n")
    f.write(f"K óptimo: {optimal_k} (selección por {K_SELECTION_TARGET})\n")
    f.write(f"Silhouette: {sil_kmeans:.4f}\n")
    f.write(f"ARI/NMI/V (Building): {ari_kmeans_b:.4f} / {nmi_kmeans_b:.4f} / {vmeas_kmeans_b:.4f}\n")
    f.write(f"ARI/NMI/V (Floor):    {ari_kmeans_f:.4f} / {nmi_kmeans_f:.4f} / {vmeas_kmeans_f:.4f}\n")
    f.write(f"ARI/NMI/V (AreaID):   {ari_kmeans_a:.4f} / {nmi_kmeans_a:.4f} / {vmeas_kmeans_a:.4f}\n\n")

    f.write("DBSCAN (sin ruido)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Eps: {optimal_eps:.4f}\n")
    f.write(f"Min samples: {optimal_min_samples}\n")
    f.write(f"Clusters: {n_clusters_dbscan}\n")
    f.write(f"Ruido: {n_noise_dbscan} ({n_noise_dbscan / len(dbscan_labels) * 100:.1f}%)\n")
    f.write(f"Cobertura: {mask_dbscan.sum()}/{len(dbscan_labels)} ({mask_dbscan.sum()/len(dbscan_labels)*100:.1f}%)\n")
    f.write(f"Silhouette: {sil_dbscan:.4f}\n")
    f.write(f"ARI/NMI/V (Building): {ari_dbscan_b:.4f} / {nmi_dbscan_b:.4f} / {vmeas_dbscan_b:.4f}\n")
    f.write(f"ARI/NMI/V (Floor):    {ari_dbscan_f:.4f} / {nmi_dbscan_f:.4f} / {vmeas_dbscan_f:.4f}\n")
    f.write(f"ARI/NMI/V (AreaID):   {ari_dbscan_a:.4f} / {nmi_dbscan_a:.4f} / {vmeas_dbscan_a:.4f}\n\n")

    f.write("CONCLUSIÓN\n")
    f.write("-" * 80 + "\n")
    winner_area = "K-Means" if nmi_kmeans_a >= nmi_dbscan_a else "DBSCAN"
    f.write(f"Algoritmo recomendado (por NMI AreaID): {winner_area}\n")

print("✓ Guardado: resultados_final.txt")
