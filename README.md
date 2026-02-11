#  Proyecto 3 — Clustering No Supervisado con WiFi Fingerprinting



##  Descripción

Este proyecto presenta un análisis experimental de métodos de **aprendizaje no supervisado** aplicados a datos de localización indoor mediante **WiFi fingerprinting (RSSI)**.

Se emplea el dataset **UJIIndoorLoc**, explorando la estructura de datos de alta dimensionalidad mediante:

- Reducción de dimensionalidad (PCA)
- Clustering (K-Means, DBSCAN)
- Evaluación con métricas internas y externas
- Visualización con t-SNE

El objetivo principal es analizar si los algoritmos de clustering pueden recuperar información espacial relevante (pisos/edificios) sin utilizar directamente las etiquetas.

---

##  Dataset

Los datos se obtuvieron del **UCI Machine Learning Repository**:

https://archive.ics.uci.edu/dataset/310/ujiindoorloc

### Archivos utilizados

Después de descargar y extraer el dataset:

- `trainingData.csv`
- `validationData.csv`

Colócalos en el directorio raíz o en la carpeta `/data`.

---

## Métodos implementados

✔ Preprocesamiento de señales RSSI  
✔ Reemplazo de valores sin señal  
✔ Eliminación de APs no informativos  
✔ Ingeniería de características estadísticas  
✔ Estandarización de datos  
✔ PCA (95% varianza)  
✔ K-Means (barrido de k)  
✔ DBSCAN (barrido de ε y MinPts)  
✔ Métricas internas y externas  
✔ Matrices de confusión  
✔ Visualización t-SNE  

---

##  Métricas de evaluación

**Internas:**

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

**Externas:**

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity
- Completeness
- V-Measure
- Purity
- Noise Ratio (DBSCAN)

---

##  Ejecución

###  Instalar dependencias

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
