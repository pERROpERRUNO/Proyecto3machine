#  Proyecto 3 — Clustering No Supervisado con WiFi Fingerprinting

## Autores

- **Meza Leon, Ricardo Manuel**
- **Ramos Bonilla, Miguel Angel**
- **Cabezas Ramirez, Dylan Andres**
- **Trujillo Flores, Frans Josep**

**Institución**: Universidad de Ingenieria y Tecnología 
**Curso**: Machine Learning 
**Fecha**: Febrero 2024

---

##  Descripción

Este proyecto implementa un **análisis exhaustivo de técnicas de clustering no supervisado** aplicadas al problema de **posicionamiento indoor mediante WiFi fingerprinting**. Utilizando señales RSSI (Received Signal Strength Indicator) de 520 puntos de acceso WiFi, se exploran dos algoritmos principales:

- **K-Means**: Para identificar estructura macro (edificios)
- **DBSCAN**: Para capturar granularidad fina (espacios específicos)

###  Objetivos Principales

1. **Análisis exploratorio** de 19,937 mediciones WiFi en entorno multipisos
2. **Reducción de dimensionalidad** mediante PCA y visualización con t-SNE
3. **Clustering jerárquico** multi-escala (edificios → pisos → áreas)
4. **Evaluación comparativa** con métricas internas y externas
5. **Optimización de hiperparámetros** para K-Means y DBSCAN

## Dataset

### Fuente de Datos

**UJIIndoorLoc Database** - UCI Machine Learning Repository  
🔗 [https://archive.ics.uci.edu/dataset/310/ujiindoorloc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)

### Características del Dataset

- **Muestras**: 19,937 (entrenamiento) + 1,111 (validación)
- **Características**: 520 WAPs (WiFi Access Points)
- **Rango RSSI**: [-104, 0] dBm (100 = señal no detectada)
- **Estructura espacial**:
  - 3 edificios
  - 5 pisos por edificio
  - 735 áreas únicas (combinación Building-Floor-Space)
- **Metadatos**: Coordenadas GPS, timestamps, IDs de usuario y dispositivo

### 📥 Descarga e Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/wifi-clustering.git
cd wifi-clustering

# Descargar dataset
wget https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip
unzip ujiindoorloc.zip

# Estructura esperada
proyecto/
├── trainingData.csv
├── validationData.csv
├── clustering_wifi_final.py
└── README.md
```

---

##  Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- 4 GB RAM mínimo (8 GB recomendado)
- 500 MB de espacio en disco

### Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

**requirements.txt:**
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

O instalar manualmente:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## Uso Rápido

### Ejecución Completa

```bash
python clustering_wifi_final.py
```

## Métodos implementados

- Preprocesamiento de señales RSSI  
- Reemplazo de valores sin señal  
- Eliminación de APs no informativos  
- Ingeniería de características estadísticas  
- Estandarización de datos  
- PCA (95% varianza)  
- K-Means (barrido de k)  
- DBSCAN (barrido de ε y MinPts)  
- Métricas internas y externas  
- Matrices de confusión  
- Visualización t-SNE  

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

