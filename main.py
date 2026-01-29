# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import time
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Загрузка датасета load_digits
digits = datasets.load_digits()

# Масштабирование признаков
X_scaled = scale(digits.data)

print("Данные загружены и отмасштабированы")
print("=" * 50)

# Вывод информации о данных
print("Размерность данных (объекты x признаки):", digits.data.shape)
print("Количество признаков:", digits.data.shape[1])
print("Количество объектов:", digits.data.shape[0])

# Количество уникальных значений в target
unique_targets = np.unique(digits.target)
n_clusters = len(unique_targets)
print("Количество уникальных значений в target:", n_clusters)
print("Уникальные значения:", unique_targets)
print("=" * 50)

# Создание модели KMeans с инициализацией k-means++
start_time = time.time()

kmeans_pp = KMeans(
    init='k-means++',  # Умная инициализация центроидов
    n_clusters=n_clusters,  # Количество кластеров равно количеству классов
    n_init=10,  # Количество запусков с разными начальными центрами
    random_state=42  # Для воспроизводимости
)

# Обучение модели
kmeans_pp.fit(X_scaled)

# Время работы
kmeans_pp_time = time.time() - start_time

# Получение меток кластеров
labels_pp = kmeans_pp.labels_

# Вычисление метрик
ari_pp = adjusted_rand_score(digits.target, labels_pp)
ami_pp = adjusted_mutual_info_score(digits.target, labels_pp)

print("Результаты для KMeans с init='k-means++':")
print(f"Время работы: {kmeans_pp_time:.4f} секунд")
print(f"Adjusted Rand Index (ARI): {ari_pp:.4f}")
print(f"Adjusted Mutual Information (AMI): {ami_pp:.4f}")
print("=" * 50)