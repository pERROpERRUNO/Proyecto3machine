"""
ESTRUCTURA:
1. Importación de librerías
2. Carga y preprocesamiento de datos
3. Análisis exploratorio
4. Reducción de dimensionalidad (PCA y t-SNE)
5. Clustering K-Means
6. Clustering DBSCAN
7. Evaluación y comparación
8. Visualizaciones finales
=================================================================================
"""

# =================================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score,
                             completeness_score, v_measure_score)
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("="*80)
print("PROYECTO: ANÁLISIS DE CLUSTERING PARA POSICIONAMIENTO INDOOR WiFi")
print("="*80)


# =================================================================================
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# =================================================================================

print("\n" + "="*80)
print("FASE 1: CARGA Y PREPROCESAMIENTO DE DATOS")
print("="*80)

# Cargar datasets
print("\n[1.1] Cargando datasets...")
df_train = pd.read_csv('trainingData.csv')
df_val = pd.read_csv('validationData.csv')

print(f"✓ Training data: {df_train.shape}")
print(f"✓ Validation data: {df_val.shape}")

# Información del dataset
print("\n[1.2] Estructura del dataset:")
print(f"- Columnas totales: {df_train.shape[1]}")
print(f"- Primeras columnas (WAPs): {[col for col in df_train.columns[:5]]}")
print(f"- Últimas columnas (metadata): {[col for col in df_train.columns[-10:]]}")

# Identificar columnas WAP (Wireless Access Points)
wap_columns = [col for col in df_train.columns if col.startswith('WAP')]
print(f"\n✓ WAPs detectados: {len(wap_columns)}")

# Separar features (señales WiFi) y targets (ubicación)
X_train = df_train[wap_columns].copy()
X_val = df_val[wap_columns].copy()

# Las etiquetas de ubicación (para evaluación externa)
# BUILDINGID, FLOOR, SPACEID pueden usarse como ground truth
y_train_building = df_train['BUILDINGID'].values
y_train_floor = df_train['FLOOR'].values
y_val_building = df_val['BUILDINGID'].values
y_val_floor = df_val['FLOOR'].values

print("\n[1.3] Preprocesamiento de señales WiFi...")

# En el dataset UJIIndoorLoc, valor 100 significa "no detectado"
# Reemplazar 100 por un valor que indique ausencia de señal
print("- Tratando valores 100 (señal no detectada)...")
X_train_processed = X_train.replace(100, -110)  # Valor muy bajo de señal
X_val_processed = X_val.replace(100, -110)

# Estadísticas antes del preprocesamiento
print("\n[1.4] Estadísticas de señales WiFi (Training):")
print(f"- Rango original: [{X_train.min().min()}, {X_train.max().max()}]")
print(f"- Media: {X_train_processed.mean().mean():.2f}")
print(f"- Desviación estándar: {X_train_processed.std().mean():.2f}")

# Eliminar WAPs con poca información (siempre ausentes)
print("\n[1.5] Filtrando WAPs con baja varianza...")
variance_threshold = 1.0
wap_variance = X_train_processed.var()
active_waps = wap_variance[wap_variance > variance_threshold].index
X_train_filtered = X_train_processed[active_waps]
X_val_filtered = X_val_processed[active_waps]

print(f"✓ WAPs activos mantenidos: {len(active_waps)} de {len(wap_columns)}")

# Normalización de datos
print("\n[1.6] Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)

print(f"✓ Datos normalizados: media=0, std=1")
print(f"- Shape final training: {X_train_scaled.shape}")
print(f"- Shape final validation: {X_val_scaled.shape}")


# =================================================================================
# 3. ANÁLISIS EXPLORATORIO
# =================================================================================

print("\n" + "="*80)
print("FASE 2: ANÁLISIS EXPLORATORIO DE DATOS")
print("="*80)

# Crear figura para análisis exploratorio
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis Exploratorio del Dataset UJIIndoorLoc', fontsize=16, fontweight='bold')

# 3.1 Distribución de edificios
ax1 = axes[0, 0]
building_counts = pd.Series(y_train_building).value_counts().sort_index()
ax1.bar(building_counts.index, building_counts.values, color='steelblue', alpha=0.7)
ax1.set_xlabel('Building ID', fontsize=12)
ax1.set_ylabel('Número de muestras', fontsize=12)
ax1.set_title('Distribución por Edificio', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 3.2 Distribución de pisos
ax2 = axes[0, 1]
floor_counts = pd.Series(y_train_floor).value_counts().sort_index()
ax2.bar(floor_counts.index, floor_counts.values, color='coral', alpha=0.7)
ax2.set_xlabel('Floor ID', fontsize=12)
ax2.set_ylabel('Número de muestras', fontsize=12)
ax2.set_title('Distribución por Piso', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3.3 Heatmap de correlación (muestra de WAPs)
ax3 = axes[1, 0]
sample_waps = np.random.choice(range(X_train_scaled.shape[1]), min(20, X_train_scaled.shape[1]), replace=False)
correlation_matrix = np.corrcoef(X_train_scaled[:, sample_waps].T)
im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax3.set_title('Correlación entre WAPs (muestra)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax3)

# 3.4 Distribución de intensidad de señal
ax4 = axes[1, 1]
signal_sample = X_train_scaled.flatten()[:10000]
ax4.hist(signal_sample, bins=50, color='green', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Intensidad de señal (normalizada)', fontsize=12)
ax4.set_ylabel('Frecuencia', fontsize=12)
ax4.set_title('Distribución de Intensidad de Señal WiFi', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_analisis_exploratorio.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 01_analisis_exploratorio.png")
plt.show()


# =================================================================================
# 4. REDUCCIÓN DE DIMENSIONALIDAD
# =================================================================================

print("\n" + "="*80)
print("FASE 3: REDUCCIÓN DE DIMENSIONALIDAD")
print("="*80)

# -----------------------------------------------------------------------------
# 4.1 PCA (Principal Component Analysis)
# -----------------------------------------------------------------------------

print("\n[3.1] Aplicando PCA (Principal Component Analysis)...")

# PCA para análisis de varianza
pca_full = PCA()
pca_full.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar número de componentes para 95% de varianza
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"✓ Componentes necesarios para 95% varianza: {n_components_95}")

# PCA para visualización (2D y 3D)
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
X_val_pca_2d = pca_2d.transform(X_val_scaled)

pca_3d = PCA(n_components=3, random_state=42)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)

print(f"✓ Varianza explicada 2D: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")
print(f"  - PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
print(f"  - PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%")

# PCA óptimo para clustering (usando 95% varianza)
pca_optimal = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca_optimal.fit_transform(X_train_scaled)
X_val_pca = pca_optimal.transform(X_val_scaled)

print(f"✓ Dataset reducido PCA: {X_train_pca.shape}")

# -----------------------------------------------------------------------------
# 4.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
# -----------------------------------------------------------------------------

print("\n[3.2] Aplicando t-SNE...")
print("⚠ Nota: t-SNE puede tardar varios minutos en datasets grandes...")

# Usar una muestra para t-SNE si el dataset es muy grande
max_samples_tsne = 5000
if X_train_scaled.shape[0] > max_samples_tsne:
    print(f"- Usando muestra de {max_samples_tsne} puntos para t-SNE...")
    sample_indices = np.random.choice(X_train_scaled.shape[0], max_samples_tsne, replace=False)
    X_train_tsne_sample = X_train_scaled[sample_indices]
    y_train_building_sample = y_train_building[sample_indices]
    y_train_floor_sample = y_train_floor[sample_indices]
else:
    X_train_tsne_sample = X_train_scaled
    y_train_building_sample = y_train_building
    y_train_floor_sample = y_train_floor

# t-SNE 2D
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_train_tsne_2d = tsne_2d.fit_transform(X_train_tsne_sample)

# t-SNE 3D
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
X_train_tsne_3d = tsne_3d.fit_transform(X_train_tsne_sample)

print(f"✓ t-SNE completado: {X_train_tsne_2d.shape}")

# Visualización de reducción de dimensionalidad
fig = plt.figure(figsize=(20, 10))
fig.suptitle('Reducción de Dimensionalidad: PCA vs t-SNE', fontsize=16, fontweight='bold')

# PCA 2D - coloreado por edificio
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=y_train_building, cmap='viridis', alpha=0.6, s=10)
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('PCA 2D - Por Edificio', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Building ID')

# PCA 2D - coloreado por piso
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=y_train_floor, cmap='plasma', alpha=0.6, s=10)
ax2.set_xlabel('PC1', fontsize=11)
ax2.set_ylabel('PC2', fontsize=11)
ax2.set_title('PCA 2D - Por Piso', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Floor ID')

# PCA Varianza explicada
ax3 = plt.subplot(2, 3, 3)
ax3.plot(range(1, len(cumulative_variance[:100])+1), cumulative_variance[:100], 'b-', linewidth=2)
ax3.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
ax3.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} componentes')
ax3.set_xlabel('Número de Componentes', fontsize=11)
ax3.set_ylabel('Varianza Acumulada', fontsize=11)
ax3.set_title('PCA: Varianza Explicada Acumulada', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# t-SNE 2D - coloreado por edificio
ax4 = plt.subplot(2, 3, 4)
scatter4 = ax4.scatter(X_train_tsne_2d[:, 0], X_train_tsne_2d[:, 1],
                       c=y_train_building_sample, cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('t-SNE 1', fontsize=11)
ax4.set_ylabel('t-SNE 2', fontsize=11)
ax4.set_title('t-SNE 2D - Por Edificio', fontsize=12, fontweight='bold')
plt.colorbar(scatter4, ax=ax4, label='Building ID')

# t-SNE 2D - coloreado por piso
ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(X_train_tsne_2d[:, 0], X_train_tsne_2d[:, 1],
                       c=y_train_floor_sample, cmap='plasma', alpha=0.6, s=10)
ax5.set_xlabel('t-SNE 1', fontsize=11)
ax5.set_ylabel('t-SNE 2', fontsize=11)
ax5.set_title('t-SNE 2D - Por Piso', fontsize=12, fontweight='bold')
plt.colorbar(scatter5, ax=ax5, label='Floor ID')

# PCA 3D
ax6 = plt.subplot(2, 3, 6, projection='3d')
scatter6 = ax6.scatter(X_train_pca_3d[:, 0], X_train_pca_3d[:, 1], X_train_pca_3d[:, 2],
                       c=y_train_building, cmap='viridis', alpha=0.4, s=5)
ax6.set_xlabel('PC1', fontsize=10)
ax6.set_ylabel('PC2', fontsize=10)
ax6.set_zlabel('PC3', fontsize=10)
ax6.set_title('PCA 3D - Por Edificio', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('02_reduccion_dimensionalidad.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 02_reduccion_dimensionalidad.png")
plt.show()


# =================================================================================
# 5. CLUSTERING CON K-MEANS
# =================================================================================

print("\n" + "="*80)
print("FASE 4: CLUSTERING CON K-MEANS")
print("="*80)

# -----------------------------------------------------------------------------
# 5.1 Método del Codo (Elbow Method) para determinar K óptimo
# -----------------------------------------------------------------------------

print("\n[4.1] Determinando K óptimo con Método del Codo...")

k_range = range(2, 21)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in k_range:
    print(f"- Probando K={k}...", end='\r')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_pca)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train_pca, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_train_pca, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_train_pca, labels))

print(f"✓ Análisis completado para K={k_range.start} a K={k_range.stop-1}")

# Visualizar métricas para selección de K
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('K-Means: Selección del Número Óptimo de Clusters', fontsize=16, fontweight='bold')

# Método del codo
ax1 = axes[0, 0]
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
ax1.set_ylabel('Inercia (Within-Cluster Sum of Squares)', fontsize=12)
ax1.set_title('Método del Codo', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Silhouette Score (mayor es mejor)
ax2 = axes[0, 1]
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
ax2.axvline(x=best_k_silhouette, color='r', linestyle='--',
            label=f'Mejor K={best_k_silhouette}')
ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score (mayor = mejor)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Davies-Bouldin Score (menor es mejor)
ax3 = axes[1, 0]
ax3.plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
best_k_db = k_range[np.argmin(davies_bouldin_scores)]
ax3.axvline(x=best_k_db, color='g', linestyle='--',
            label=f'Mejor K={best_k_db}')
ax3.set_xlabel('Número de Clusters (K)', fontsize=12)
ax3.set_ylabel('Davies-Bouldin Score', fontsize=12)
ax3.set_title('Davies-Bouldin Score (menor = mejor)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Calinski-Harabasz Score (mayor es mejor)
ax4 = axes[1, 1]
ax4.plot(k_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
best_k_ch = k_range[np.argmax(calinski_harabasz_scores)]
ax4.axvline(x=best_k_ch, color='b', linestyle='--',
            label=f'Mejor K={best_k_ch}')
ax4.set_xlabel('Número de Clusters (K)', fontsize=12)
ax4.set_ylabel('Calinski-Harabasz Score', fontsize=12)
ax4.set_title('Calinski-Harabasz Score (mayor = mejor)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_kmeans_seleccion_k.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 03_kmeans_seleccion_k.png")
plt.show()

print("\n[4.2] Recomendaciones para K óptimo:")
print(f"  - Silhouette Score sugiere: K={best_k_silhouette}")
print(f"  - Davies-Bouldin sugiere: K={best_k_db}")
print(f"  - Calinski-Harabasz sugiere: K={best_k_ch}")

# Seleccionar K basado en múltiples criterios (usaremos el de mejor Silhouette)
optimal_k = best_k_silhouette
print(f"\n✓ K ÓPTIMO SELECCIONADO: {optimal_k}")

# -----------------------------------------------------------------------------
# 5.2 Aplicar K-Means con K óptimo
# -----------------------------------------------------------------------------

print(f"\n[4.3] Aplicando K-Means con K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
kmeans_labels_train = kmeans_final.fit_predict(X_train_pca)
kmeans_labels_val = kmeans_final.predict(X_val_pca)

print(f"✓ K-Means completado")
print(f"  - Clusters en training: {len(np.unique(kmeans_labels_train))}")
print(f"  - Distribución de clusters: {np.bincount(kmeans_labels_train)}")

# Visualización de clusters K-Means
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'K-Means Clustering (K={optimal_k})', fontsize=16, fontweight='bold')

# K-Means en espacio PCA 2D
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=kmeans_labels_train, cmap='tab20', alpha=0.6, s=10)
centers = kmeans_final.cluster_centers_
centers_2d = pca_2d.transform(pca_optimal.inverse_transform(centers))
ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X',
            s=200, edgecolors='black', linewidths=2, label='Centroides')
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('K-Means en Espacio PCA', fontsize=12, fontweight='bold')
ax1.legend()
plt.colorbar(scatter1, ax=ax1, label='Cluster ID')

# K-Means en espacio t-SNE
ax2 = plt.subplot(2, 3, 2)
if len(kmeans_labels_train) > max_samples_tsne:
    kmeans_labels_sample = kmeans_labels_train[sample_indices]
else:
    kmeans_labels_sample = kmeans_labels_train
scatter2 = ax2.scatter(X_train_tsne_2d[:, 0], X_train_tsne_2d[:, 1],
                       c=kmeans_labels_sample, cmap='tab20', alpha=0.6, s=10)
ax2.set_xlabel('t-SNE 1', fontsize=11)
ax2.set_ylabel('t-SNE 2', fontsize=11)
ax2.set_title('K-Means en Espacio t-SNE', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

# Distribución de tamaños de clusters
ax3 = plt.subplot(2, 3, 3)
cluster_sizes = np.bincount(kmeans_labels_train)
ax3.bar(range(len(cluster_sizes)), cluster_sizes, color='steelblue', alpha=0.7)
ax3.set_xlabel('Cluster ID', fontsize=11)
ax3.set_ylabel('Número de Puntos', fontsize=11)
ax3.set_title('Tamaño de Clusters', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Comparación con ground truth - Building
ax4 = plt.subplot(2, 3, 4)
scatter4 = ax4.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=y_train_building, cmap='viridis', alpha=0.6, s=10)
ax4.set_xlabel('PC1', fontsize=11)
ax4.set_ylabel('PC2', fontsize=11)
ax4.set_title('Ground Truth: Edificios', fontsize=12, fontweight='bold')
plt.colorbar(scatter4, ax=ax4, label='Building ID')

# Comparación con ground truth - Floor
ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=y_train_floor, cmap='plasma', alpha=0.6, s=10)
ax5.set_xlabel('PC1', fontsize=11)
ax5.set_ylabel('PC2', fontsize=11)
ax5.set_title('Ground Truth: Pisos', fontsize=12, fontweight='bold')
plt.colorbar(scatter5, ax=ax5, label='Floor ID')

# Matriz de confusión: Clusters vs Building
ax6 = plt.subplot(2, 3, 6)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train_building, kmeans_labels_train)
im = ax6.imshow(conf_matrix, cmap='YlOrRd', aspect='auto')
ax6.set_xlabel('Cluster K-Means', fontsize=11)
ax6.set_ylabel('Building ID (Ground Truth)', fontsize=11)
ax6.set_title('Matriz de Confusión: Clusters vs Edificios', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.savefig('04_kmeans_resultados.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 04_kmeans_resultados.png")
plt.show()


# =================================================================================
# 6. CLUSTERING CON DBSCAN
# =================================================================================

print("\n" + "="*80)
print("FASE 5: CLUSTERING CON DBSCAN")
print("="*80)

# -----------------------------------------------------------------------------
# 6.1 Determinar parámetros óptimos (eps y min_samples)
# -----------------------------------------------------------------------------

print("\n[5.1] Determinando parámetros óptimos para DBSCAN...")
print("⚠ Nota: Este proceso puede tardar varios minutos...")

# Calcular distancias a k-vecinos más cercanos para estimar eps
from sklearn.neighbors import NearestNeighbors

k = 4  # min_samples típicamente se establece en k = dimensionalidad + 1
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_train_pca)
distances, indices = neighbors.kneighbors(X_train_pca)

# Ordenar distancias para encontrar el "codo"
distances_sorted = np.sort(distances[:, k-1], axis=0)

# Estimar eps automáticamente (usar el percentil 90)
eps_estimate = np.percentile(distances_sorted, 90)
print(f"✓ Eps estimado: {eps_estimate:.4f}")

# Probar diferentes configuraciones
print("\n[5.2] Probando diferentes configuraciones de DBSCAN...")

eps_values = [eps_estimate * 0.5, eps_estimate, eps_estimate * 1.5]
min_samples_values = [3, 5, 10]

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        print(f"- Probando eps={eps:.4f}, min_samples={min_samples}...", end='\r')

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_train_pca)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Solo calcular métricas si hay al menos 2 clusters
        if n_clusters >= 2:
            # Filtrar ruido para métricas
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X_train_pca[mask], labels[mask])
                davies_bouldin = davies_bouldin_score(X_train_pca[mask], labels[mask])
                calinski_harabasz = calinski_harabasz_score(X_train_pca[mask], labels[mask])
            else:
                silhouette = -1
                davies_bouldin = 999
                calinski_harabasz = 0
        else:
            silhouette = -1
            davies_bouldin = 999
            calinski_harabasz = 0

        dbscan_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(labels),
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        })

print("\n✓ Análisis de parámetros DBSCAN completado")

# Convertir a DataFrame para análisis
df_dbscan = pd.DataFrame(dbscan_results)
print("\n[5.3] Resultados de configuraciones DBSCAN:")
print(df_dbscan.to_string(index=False))

# Seleccionar mejor configuración
# Filtrar configuraciones con clusters razonables (entre 3 y 30) y poco ruido (<30%)
valid_configs = df_dbscan[(df_dbscan['n_clusters'] >= 3) &
                          (df_dbscan['n_clusters'] <= 30) &
                          (df_dbscan['noise_ratio'] < 0.3)]

if len(valid_configs) > 0:
    # Seleccionar basado en mejor Silhouette Score
    best_config = valid_configs.loc[valid_configs['silhouette'].idxmax()]
else:
    # Si no hay configuraciones válidas, usar la que tenga más clusters y menos ruido
    best_config = df_dbscan.loc[(df_dbscan['n_clusters'] - df_dbscan['noise_ratio']).idxmax()]

best_eps = best_config['eps']
best_min_samples = int(best_config['min_samples'])

print(f"\n✓ PARÁMETROS ÓPTIMOS SELECCIONADOS:")
print(f"  - eps: {best_eps:.4f}")
print(f"  - min_samples: {best_min_samples}")
print(f"  - Clusters esperados: {int(best_config['n_clusters'])}")
print(f"  - Ruido esperado: {best_config['noise_ratio']*100:.2f}%")

# Visualización de selección de parámetros
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DBSCAN: Selección de Parámetros', fontsize=16, fontweight='bold')

# K-distance plot
ax1 = axes[0, 0]
ax1.plot(distances_sorted, 'b-', linewidth=1)
ax1.axhline(y=eps_estimate, color='r', linestyle='--', label=f'Eps estimado={eps_estimate:.4f}')
ax1.set_xlabel('Puntos ordenados', fontsize=12)
ax1.set_ylabel('Distancia al k-ésimo vecino más cercano', fontsize=12)
ax1.set_title('K-Distance Plot', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Número de clusters vs parámetros
ax2 = axes[0, 1]
for eps_val in eps_values:
    subset = df_dbscan[df_dbscan['eps'] == eps_val]
    ax2.plot(subset['min_samples'], subset['n_clusters'], 'o-',
             label=f'eps={eps_val:.4f}', linewidth=2, markersize=8)
ax2.set_xlabel('min_samples', fontsize=12)
ax2.set_ylabel('Número de Clusters', fontsize=12)
ax2.set_title('Clusters vs Parámetros', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Ratio de ruido vs parámetros
ax3 = axes[1, 0]
for eps_val in eps_values:
    subset = df_dbscan[df_dbscan['eps'] == eps_val]
    ax3.plot(subset['min_samples'], subset['noise_ratio']*100, 'o-',
             label=f'eps={eps_val:.4f}', linewidth=2, markersize=8)
ax3.set_xlabel('min_samples', fontsize=12)
ax3.set_ylabel('Ruido (%)', fontsize=12)
ax3.set_title('Porcentaje de Ruido vs Parámetros', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Silhouette Score vs parámetros
ax4 = axes[1, 1]
for eps_val in eps_values:
    subset = df_dbscan[df_dbscan['eps'] == eps_val]
    subset_valid = subset[subset['silhouette'] > -1]
    if len(subset_valid) > 0:
        ax4.plot(subset_valid['min_samples'], subset_valid['silhouette'], 'o-',
                 label=f'eps={eps_val:.4f}', linewidth=2, markersize=8)
ax4.set_xlabel('min_samples', fontsize=12)
ax4.set_ylabel('Silhouette Score', fontsize=12)
ax4.set_title('Silhouette Score vs Parámetros', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_dbscan_seleccion_parametros.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 05_dbscan_seleccion_parametros.png")
plt.show()

# -----------------------------------------------------------------------------
# 6.2 Aplicar DBSCAN con parámetros óptimos
# -----------------------------------------------------------------------------

print(f"\n[5.4] Aplicando DBSCAN con parámetros óptimos...")

dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels_train = dbscan_final.fit_predict(X_train_pca)

n_clusters_dbscan = len(set(dbscan_labels_train)) - (1 if -1 in dbscan_labels_train else 0)
n_noise_dbscan = list(dbscan_labels_train).count(-1)

print(f"✓ DBSCAN completado")
print(f"  - Clusters encontrados: {n_clusters_dbscan}")
print(f"  - Puntos de ruido: {n_noise_dbscan} ({n_noise_dbscan/len(dbscan_labels_train)*100:.2f}%)")
print(f"  - Distribución de clusters (sin ruido):")

cluster_counts = np.bincount(dbscan_labels_train[dbscan_labels_train != -1])
for i, count in enumerate(cluster_counts):
    print(f"    Cluster {i}: {count} puntos")

# Visualización de resultados DBSCAN
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'DBSCAN Clustering (eps={best_eps:.4f}, min_samples={best_min_samples})',
             fontsize=16, fontweight='bold')

# DBSCAN en espacio PCA 2D
ax1 = plt.subplot(2, 3, 1)
# Separar ruido de clusters
core_mask = dbscan_labels_train != -1
scatter1 = ax1.scatter(X_train_pca_2d[core_mask, 0], X_train_pca_2d[core_mask, 1],
                       c=dbscan_labels_train[core_mask], cmap='tab20', alpha=0.6, s=10,
                       label='Clusters')
ax1.scatter(X_train_pca_2d[~core_mask, 0], X_train_pca_2d[~core_mask, 1],
            c='gray', alpha=0.3, s=5, marker='x', label='Ruido')
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('DBSCAN en Espacio PCA', fontsize=12, fontweight='bold')
ax1.legend()
plt.colorbar(scatter1, ax=ax1, label='Cluster ID')

# DBSCAN en espacio t-SNE
ax2 = plt.subplot(2, 3, 2)
if len(dbscan_labels_train) > max_samples_tsne:
    dbscan_labels_sample = dbscan_labels_train[sample_indices]
    core_mask_sample = dbscan_labels_sample != -1
else:
    dbscan_labels_sample = dbscan_labels_train
    core_mask_sample = core_mask

scatter2 = ax2.scatter(X_train_tsne_2d[core_mask_sample, 0],
                       X_train_tsne_2d[core_mask_sample, 1],
                       c=dbscan_labels_sample[core_mask_sample],
                       cmap='tab20', alpha=0.6, s=10, label='Clusters')
ax2.scatter(X_train_tsne_2d[~core_mask_sample, 0],
            X_train_tsne_2d[~core_mask_sample, 1],
            c='gray', alpha=0.3, s=5, marker='x', label='Ruido')
ax2.set_xlabel('t-SNE 1', fontsize=11)
ax2.set_ylabel('t-SNE 2', fontsize=11)
ax2.set_title('DBSCAN en Espacio t-SNE', fontsize=12, fontweight='bold')
ax2.legend()
plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

# Distribución de tamaños de clusters
ax3 = plt.subplot(2, 3, 3)
if n_clusters_dbscan > 0:
    cluster_sizes_db = np.bincount(dbscan_labels_train[dbscan_labels_train != -1])
    ax3.bar(range(len(cluster_sizes_db)), cluster_sizes_db, color='teal', alpha=0.7)
    ax3.set_xlabel('Cluster ID', fontsize=11)
    ax3.set_ylabel('Número de Puntos', fontsize=11)
    ax3.set_title(f'Tamaño de Clusters (Ruido: {n_noise_dbscan} pts)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

# Densidad de puntos
ax4 = plt.subplot(2, 3, 4)
hex_plot = ax4.hexbin(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                      gridsize=30, cmap='YlOrRd', mincnt=1)
ax4.set_xlabel('PC1', fontsize=11)
ax4.set_ylabel('PC2', fontsize=11)
ax4.set_title('Densidad de Puntos (Hexbin)', fontsize=12, fontweight='bold')
plt.colorbar(hex_plot, ax=ax4, label='Densidad')

# Comparación con K-Means
ax5 = plt.subplot(2, 3, 5)
scatter5 = ax5.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
                       c=kmeans_labels_train, cmap='tab20', alpha=0.6, s=10)
ax5.set_xlabel('PC1', fontsize=11)
ax5.set_ylabel('PC2', fontsize=11)
ax5.set_title('Comparación: K-Means', fontsize=12, fontweight='bold')
plt.colorbar(scatter5, ax=ax5, label='Cluster ID')

# Matriz de confusión: Clusters vs Building (sin ruido)
ax6 = plt.subplot(2, 3, 6)
if n_clusters_dbscan > 0:
    mask_no_noise = dbscan_labels_train != -1
    conf_matrix_db = confusion_matrix(y_train_building[mask_no_noise],
                                      dbscan_labels_train[mask_no_noise])
    im = ax6.imshow(conf_matrix_db, cmap='YlOrRd', aspect='auto')
    ax6.set_xlabel('Cluster DBSCAN', fontsize=11)
    ax6.set_ylabel('Building ID (Ground Truth)', fontsize=11)
    ax6.set_title('Matriz de Confusión: Clusters vs Edificios', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.savefig('06_dbscan_resultados.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 06_dbscan_resultados.png")
plt.show()


# =================================================================================
# 7. EVALUACIÓN Y COMPARACIÓN DE ALGORITMOS
# =================================================================================

print("\n" + "="*80)
print("FASE 6: EVALUACIÓN Y COMPARACIÓN DE ALGORITMOS")
print("="*80)

# -----------------------------------------------------------------------------
# 7.1 Evaluación Interna (sin usar etiquetas)
# -----------------------------------------------------------------------------

print("\n[6.1] EVALUACIÓN INTERNA (Métricas sin etiquetas verdaderas)")
print("-" * 80)

# K-Means
kmeans_mask = np.ones(len(kmeans_labels_train), dtype=bool)
kmeans_silhouette = silhouette_score(X_train_pca, kmeans_labels_train)
kmeans_davies_bouldin = davies_bouldin_score(X_train_pca, kmeans_labels_train)
kmeans_calinski = calinski_harabasz_score(X_train_pca, kmeans_labels_train)

print("\n📊 K-MEANS:")
print(f"  ✓ Silhouette Score: {kmeans_silhouette:.4f} (rango: -1 a 1, mayor es mejor)")
print(f"  ✓ Davies-Bouldin Index: {kmeans_davies_bouldin:.4f} (menor es mejor)")
print(f"  ✓ Calinski-Harabasz Index: {kmeans_calinski:.2f} (mayor es mejor)")

# DBSCAN (filtrar ruido)
dbscan_mask = dbscan_labels_train != -1
if dbscan_mask.sum() > 0 and n_clusters_dbscan >= 2:
    dbscan_silhouette = silhouette_score(X_train_pca[dbscan_mask],
                                         dbscan_labels_train[dbscan_mask])
    dbscan_davies_bouldin = davies_bouldin_score(X_train_pca[dbscan_mask],
                                                  dbscan_labels_train[dbscan_mask])
    dbscan_calinski = calinski_harabasz_score(X_train_pca[dbscan_mask],
                                               dbscan_labels_train[dbscan_mask])

    print("\n📊 DBSCAN:")
    print(f"  ✓ Silhouette Score: {dbscan_silhouette:.4f} (rango: -1 a 1, mayor es mejor)")
    print(f"  ✓ Davies-Bouldin Index: {dbscan_davies_bouldin:.4f} (menor es mejor)")
    print(f"  ✓ Calinski-Harabasz Index: {dbscan_calinski:.2f} (mayor es mejor)")
    print(f"  ⚠ Puntos clasificados: {dbscan_mask.sum()}/{len(dbscan_labels_train)} " +
          f"({dbscan_mask.sum()/len(dbscan_labels_train)*100:.1f}%)")
else:
    dbscan_silhouette = -1
    dbscan_davies_bouldin = 999
    dbscan_calinski = 0
    print("\n📊 DBSCAN:")
    print("  ⚠ No se pueden calcular métricas (muy pocos clusters o muchos puntos de ruido)")

# -----------------------------------------------------------------------------
# 7.2 Evaluación Externa (usando etiquetas verdaderas)
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("[6.2] EVALUACIÓN EXTERNA (Comparación con ground truth)")
print("-" * 80)

# Evaluar contra Building ID
print("\n🏢 COMPARACIÓN CON BUILDING ID:")

# K-Means vs Building
kmeans_ari_building = adjusted_rand_score(y_train_building, kmeans_labels_train)
kmeans_nmi_building = normalized_mutual_info_score(y_train_building, kmeans_labels_train)
kmeans_homogeneity_building = homogeneity_score(y_train_building, kmeans_labels_train)
kmeans_completeness_building = completeness_score(y_train_building, kmeans_labels_train)
kmeans_v_measure_building = v_measure_score(y_train_building, kmeans_labels_train)

print("\n  K-Means:")
print(f"    • Adjusted Rand Index (ARI): {kmeans_ari_building:.4f}")
print(f"    • Normalized Mutual Info (NMI): {kmeans_nmi_building:.4f}")
print(f"    • Homogeneity: {kmeans_homogeneity_building:.4f}")
print(f"    • Completeness: {kmeans_completeness_building:.4f}")
print(f"    • V-Measure: {kmeans_v_measure_building:.4f}")

# DBSCAN vs Building (sin ruido)
if dbscan_mask.sum() > 0:
    dbscan_ari_building = adjusted_rand_score(y_train_building[dbscan_mask],
                                               dbscan_labels_train[dbscan_mask])
    dbscan_nmi_building = normalized_mutual_info_score(y_train_building[dbscan_mask],
                                                        dbscan_labels_train[dbscan_mask])
    dbscan_homogeneity_building = homogeneity_score(y_train_building[dbscan_mask],
                                                     dbscan_labels_train[dbscan_mask])
    dbscan_completeness_building = completeness_score(y_train_building[dbscan_mask],
                                                       dbscan_labels_train[dbscan_mask])
    dbscan_v_measure_building = v_measure_score(y_train_building[dbscan_mask],
                                                 dbscan_labels_train[dbscan_mask])

    print("\n  DBSCAN:")
    print(f"    • Adjusted Rand Index (ARI): {dbscan_ari_building:.4f}")
    print(f"    • Normalized Mutual Info (NMI): {dbscan_nmi_building:.4f}")
    print(f"    • Homogeneity: {dbscan_homogeneity_building:.4f}")
    print(f"    • Completeness: {dbscan_completeness_building:.4f}")
    print(f"    • V-Measure: {dbscan_v_measure_building:.4f}")
else:
    dbscan_ari_building = 0
    dbscan_nmi_building = 0
    dbscan_homogeneity_building = 0
    dbscan_completeness_building = 0
    dbscan_v_measure_building = 0

# Evaluar contra Floor ID
print("\n🏢 COMPARACIÓN CON FLOOR ID:")

# K-Means vs Floor
kmeans_ari_floor = adjusted_rand_score(y_train_floor, kmeans_labels_train)
kmeans_nmi_floor = normalized_mutual_info_score(y_train_floor, kmeans_labels_train)
kmeans_v_measure_floor = v_measure_score(y_train_floor, kmeans_labels_train)

print("\n  K-Means:")
print(f"    • Adjusted Rand Index (ARI): {kmeans_ari_floor:.4f}")
print(f"    • Normalized Mutual Info (NMI): {kmeans_nmi_floor:.4f}")
print(f"    • V-Measure: {kmeans_v_measure_floor:.4f}")

# DBSCAN vs Floor
if dbscan_mask.sum() > 0:
    dbscan_ari_floor = adjusted_rand_score(y_train_floor[dbscan_mask],
                                           dbscan_labels_train[dbscan_mask])
    dbscan_nmi_floor = normalized_mutual_info_score(y_train_floor[dbscan_mask],
                                                     dbscan_labels_train[dbscan_mask])
    dbscan_v_measure_floor = v_measure_score(y_train_floor[dbscan_mask],
                                              dbscan_labels_train[dbscan_mask])

    print("\n  DBSCAN:")
    print(f"    • Adjusted Rand Index (ARI): {dbscan_ari_floor:.4f}")
    print(f"    • Normalized Mutual Info (NMI): {dbscan_nmi_floor:.4f}")
    print(f"    • V-Measure: {dbscan_v_measure_floor:.4f}")
else:
    dbscan_ari_floor = 0
    dbscan_nmi_floor = 0
    dbscan_v_measure_floor = 0

# -----------------------------------------------------------------------------
# 7.3 Tabla comparativa
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("[6.3] TABLA COMPARATIVA COMPLETA")
print("-" * 80)

comparison_data = {
    'Métrica': [
        'Silhouette Score ↑',
        'Davies-Bouldin ↓',
        'Calinski-Harabasz ↑',
        'ARI (Building) ↑',
        'NMI (Building) ↑',
        'V-Measure (Building) ↑',
        'ARI (Floor) ↑',
        'NMI (Floor) ↑',
        'V-Measure (Floor) ↑',
        'Número de Clusters',
        'Puntos de Ruido'
    ],
    'K-Means': [
        f'{kmeans_silhouette:.4f}',
        f'{kmeans_davies_bouldin:.4f}',
        f'{kmeans_calinski:.2f}',
        f'{kmeans_ari_building:.4f}',
        f'{kmeans_nmi_building:.4f}',
        f'{kmeans_v_measure_building:.4f}',
        f'{kmeans_ari_floor:.4f}',
        f'{kmeans_nmi_floor:.4f}',
        f'{kmeans_v_measure_floor:.4f}',
        f'{optimal_k}',
        '0'
    ],
    'DBSCAN': [
        f'{dbscan_silhouette:.4f}' if dbscan_silhouette > -1 else 'N/A',
        f'{dbscan_davies_bouldin:.4f}' if dbscan_davies_bouldin < 999 else 'N/A',
        f'{dbscan_calinski:.2f}' if dbscan_calinski > 0 else 'N/A',
        f'{dbscan_ari_building:.4f}',
        f'{dbscan_nmi_building:.4f}',
        f'{dbscan_v_measure_building:.4f}',
        f'{dbscan_ari_floor:.4f}',
        f'{dbscan_nmi_floor:.4f}',
        f'{dbscan_v_measure_floor:.4f}',
        f'{n_clusters_dbscan}',
        f'{n_noise_dbscan}'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Visualización comparativa
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparación de Algoritmos: K-Means vs DBSCAN', fontsize=16, fontweight='bold')

# Métricas internas
ax1 = axes[0, 0]
metrics_internal = ['Silhouette', 'Davies-Bouldin\n(invertido)', 'Calinski-Harabasz\n(normalizado)']
kmeans_internal = [kmeans_silhouette, 1/kmeans_davies_bouldin, kmeans_calinski/1000]
dbscan_internal = [dbscan_silhouette if dbscan_silhouette > -1 else 0,
                   1/dbscan_davies_bouldin if dbscan_davies_bouldin < 999 else 0,
                   dbscan_calinski/1000]
x_pos = np.arange(len(metrics_internal))
width = 0.35
ax1.bar(x_pos - width/2, kmeans_internal, width, label='K-Means', color='steelblue', alpha=0.8)
ax1.bar(x_pos + width/2, dbscan_internal, width, label='DBSCAN', color='teal', alpha=0.8)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Evaluación Interna (sin etiquetas)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics_internal, fontsize=9)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Métricas externas - Building
ax2 = axes[0, 1]
metrics_external_build = ['ARI', 'NMI', 'V-Measure']
kmeans_external_build = [kmeans_ari_building, kmeans_nmi_building, kmeans_v_measure_building]
dbscan_external_build = [dbscan_ari_building, dbscan_nmi_building, dbscan_v_measure_building]
x_pos = np.arange(len(metrics_external_build))
ax2.bar(x_pos - width/2, kmeans_external_build, width, label='K-Means', color='coral', alpha=0.8)
ax2.bar(x_pos + width/2, dbscan_external_build, width, label='DBSCAN', color='salmon', alpha=0.8)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Evaluación Externa: Building ID', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_external_build, fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1])

# Métricas externas - Floor
ax3 = axes[0, 2]
metrics_external_floor = ['ARI', 'NMI', 'V-Measure']
kmeans_external_floor = [kmeans_ari_floor, kmeans_nmi_floor, kmeans_v_measure_floor]
dbscan_external_floor = [dbscan_ari_floor, dbscan_nmi_floor, dbscan_v_measure_floor]
x_pos = np.arange(len(metrics_external_floor))
ax3.bar(x_pos - width/2, kmeans_external_floor, width, label='K-Means', color='purple', alpha=0.8)
ax3.bar(x_pos + width/2, dbscan_external_floor, width, label='DBSCAN', color='violet', alpha=0.8)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Evaluación Externa: Floor ID', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics_external_floor, fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim([0, 1])

# Distribución de tamaños de cluster - K-Means
ax4 = axes[1, 0]
ax4.hist(kmeans_labels_train, bins=optimal_k, color='steelblue', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Cluster ID', fontsize=11)
ax4.set_ylabel('Frecuencia', fontsize=11)
ax4.set_title(f'K-Means: Distribución de Clusters (K={optimal_k})',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Distribución de tamaños de cluster - DBSCAN
ax5 = axes[1, 1]
dbscan_no_noise = dbscan_labels_train[dbscan_labels_train != -1]
if len(dbscan_no_noise) > 0:
    ax5.hist(dbscan_no_noise, bins=n_clusters_dbscan, color='teal', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Cluster ID', fontsize=11)
ax5.set_ylabel('Frecuencia', fontsize=11)
ax5.set_title(f'DBSCAN: Distribución de Clusters (n={n_clusters_dbscan}, ruido={n_noise_dbscan})',
              fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Comparación lado a lado en PCA
ax6 = axes[1, 2]
ax6.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1],
            c=kmeans_labels_train, cmap='tab20', alpha=0.3, s=5, label='K-Means')
core_mask_plot = dbscan_labels_train != -1
ax6.scatter(X_train_pca_2d[core_mask_plot, 0], X_train_pca_2d[core_mask_plot, 1],
            c=dbscan_labels_train[core_mask_plot], cmap='Set1', alpha=0.6, s=5,
            marker='^', label='DBSCAN')
ax6.set_xlabel('PC1', fontsize=11)
ax6.set_ylabel('PC2', fontsize=11)
ax6.set_title('Superposición de Clusters', fontsize=12, fontweight='bold')
ax6.legend()

plt.tight_layout()
plt.savefig('07_comparacion_algoritmos.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 07_comparacion_algoritmos.png")
plt.show()


# =================================================================================
# 8. RECOMENDACIONES Y CONCLUSIONES
# =================================================================================

print("\n" + "="*80)
print("FASE 7: RECOMENDACIONES Y CONCLUSIONES")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ANÁLISIS Y RECOMENDACIONES FINALES                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 RESUMEN DE RESULTADOS:
""")
dbscan_silhouette_str = f"{dbscan_silhouette:.4f}" if dbscan_silhouette > -1 else "N/A"

print(f"""
1. PREPROCESAMIENTO:
   ✓ Dataset procesado: {X_train_scaled.shape[0]} muestras, {X_train_scaled.shape[1]} WAPs activos
   ✓ Reducción PCA: {n_components_95} componentes para 95% varianza
   ✓ Ground truth disponible: {len(np.unique(y_train_building))} edificios, {len(np.unique(y_train_floor))} pisos

2. K-MEANS (K={optimal_k}):
   • Silhouette Score: {kmeans_silhouette:.4f}
   • ARI (Building): {kmeans_ari_building:.4f}
   • NMI (Building): {kmeans_nmi_building:.4f}
   • Ventajas: Clustering balanceado, rápido, interpretable
   • Desventajas: Asume clusters esféricos, requiere K predefinido

3. DBSCAN (eps={best_eps:.4f}, min_samples={best_min_samples}):
   • Clusters encontrados: {n_clusters_dbscan}
   • Puntos de ruido: {n_noise_dbscan} ({n_noise_dbscan/len(dbscan_labels_train)*100:.1f}%)
   • Silhouette Score: {dbscan_silhouette_str}
   • ARI (Building): {dbscan_ari_building:.4f}
   • Ventajas: Detecta formas arbitrarias, identifica outliers
   • Desventajas: Sensible a parámetros, puede generar mucho ruido
""")


# Guardar resultados en archivo
print("\n[7.1] Guardando resultados finales...")

with open('resultados_clustering.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RESULTADOS DEL ANÁLISIS DE CLUSTERING - UJIIndoorLoc\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET:\n")
    f.write(f"  - Training samples: {X_train_scaled.shape[0]}\n")
    f.write(f"  - Validation samples: {X_val_scaled.shape[0]}\n")
    f.write(f"  - WAPs activos: {X_train_scaled.shape[1]}\n")
    f.write(f"  - Componentes PCA (95%): {n_components_95}\n\n")

    f.write("K-MEANS:\n")
    f.write(f"  - K óptimo: {optimal_k}\n")
    f.write(f"  - Silhouette Score: {kmeans_silhouette:.4f}\n")
    f.write(f"  - Davies-Bouldin: {kmeans_davies_bouldin:.4f}\n")
    f.write(f"  - Calinski-Harabasz: {kmeans_calinski:.2f}\n")
    f.write(f"  - ARI (Building): {kmeans_ari_building:.4f}\n")
    f.write(f"  - NMI (Building): {kmeans_nmi_building:.4f}\n\n")

    f.write("DBSCAN:\n")
    f.write(f"  - eps: {best_eps:.4f}\n")
    f.write(f"  - min_samples: {best_min_samples}\n")
    f.write(f"  - Clusters: {n_clusters_dbscan}\n")
    f.write(f"  - Ruido: {n_noise_dbscan} ({n_noise_dbscan/len(dbscan_labels_train)*100:.1f}%)\n")
    if dbscan_silhouette > -1:
        f.write(f"  - Silhouette Score: {dbscan_silhouette:.4f}\n")
        f.write(f"  - Davies-Bouldin: {dbscan_davies_bouldin:.4f}\n")
        f.write(f"  - Calinski-Harabasz: {dbscan_calinski:.2f}\n")
    f.write(f"  - ARI (Building): {dbscan_ari_building:.4f}\n")
    f.write(f"  - NMI (Building): {dbscan_nmi_building:.4f}\n\n")

    f.write("COMPARACIÓN:\n")
    f.write(df_comparison.to_string(index=False))
    f.write("\n\n")

print("✓ Resultados guardados en: resultados_clustering.txt")

print("\n" + "="*80)
print("PROYECTO COMPLETADO EXITOSAMENTE")
print("="*80)
