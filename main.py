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

# Метод локтя для определения оптимального числа кластеров
# (для сравнения с заданным n_clusters)
inertia_values = []
k_range = range(2, 20)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertia_values.append(kmeans_temp.inertia_)

# Метод силуэта
silhouette_scores = []
for k in k_range:
    if k < len(X_scaled):  # Проверка, чтобы k было меньше количества объектов
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans_temp.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

# Визуализация метода локтя
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia_values, 'bo-')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia (сумма квадратов расстояний)')
plt.title('Метод локтя для KMeans (k-means++)')
plt.axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters} (кол-во классов)')
plt.grid(True, alpha=0.3)
plt.legend()

# Визуализация метода силуэта
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Средний коэффициент силуэта')
plt.title('Метод силуэта для KMeans (k-means++)')
plt.axvline(x=n_clusters, color='r', linestyle='--', label=f'k={n_clusters} (кол-во классов)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Коэффициент силуэта для k={n_clusters}: {silhouette_scores[n_clusters-2]:.4f}")
print("=" * 50)

# Создание модели KMeans с инициализацией random
start_time = time.time()

kmeans_random = KMeans(
    init='random',  # Случайная инициализация центроидов
    n_clusters=n_clusters,
    n_init=10,
    random_state=42
)

# Обучение модели
kmeans_random.fit(X_scaled)

# Время работы
kmeans_random_time = time.time() - start_time

# Получение меток кластеров
labels_random = kmeans_random.labels_

# Вычисление метрик
ari_random = adjusted_rand_score(digits.target, labels_random)
ami_random = adjusted_mutual_info_score(digits.target, labels_random)

print("Результаты для KMeans с init='random':")
print(f"Время работы: {kmeans_random_time:.4f} секунд")
print(f"Adjusted Rand Index (ARI): {ari_random:.4f}")
print(f"Adjusted Mutual Information (AMI): {ami_random:.4f}")
print("=" * 50)

# Методы локтя и силуэта для random
inertia_values_random = []
silhouette_scores_random = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertia_values_random.append(kmeans_temp.inertia_)
    
    if k < len(X_scaled):
        cluster_labels = kmeans_temp.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores_random.append(silhouette_avg)

# Применение метода PCA
pca = PCA(n_components=n_clusters)  # Количество компонент равно количеству уникальных классов
X_pca = pca.fit_transform(X_scaled)

print("Результаты PCA:")
print(f"Объясненная дисперсия для каждой компоненты: {pca.explained_variance_ratio_}")
print(f"Суммарная объясненная дисперсия: {sum(pca.explained_variance_ratio_):.4f}")
print(f"Собственные значения: {pca.explained_variance_}")
print("=" * 50)

# Создание модели KMeans с инициализацией из компонент PCA
start_time = time.time()

kmeans_pca = KMeans(
    init=pca.components_[:n_clusters],  # Используем компоненты PCA как начальные центры
    n_clusters=n_clusters,
    n_init=1,  # Только одна инициализация, т.к. мы задали центры вручную
    random_state=42
)

# Обучение модели
kmeans_pca.fit(X_scaled)

# Время работы
kmeans_pca_time = time.time() - start_time

# Получение меток кластеров
labels_pca = kmeans_pca.labels_

# Вычисление метрик
ari_pca = adjusted_rand_score(digits.target, labels_pca)
ami_pca = adjusted_mutual_info_score(digits.target, labels_pca)

print("Результаты для KMeans с init из PCA компонент:")
print(f"Время работы: {kmeans_pca_time:.4f} секунд")
print(f"Adjusted Rand Index (ARI): {ari_pca:.4f}")
print(f"Adjusted Mutual Information (AMI): {ami_pca:.4f}")
print("=" * 50)


# Создание таблицы сравнения
comparison_df = pd.DataFrame({
    'Метод инициализации': ['k-means++', 'random', 'PCA компоненты'],
    'Время (сек)': [kmeans_pp_time, kmeans_random_time, kmeans_pca_time],
    'ARI': [ari_pp, ari_random, ari_pca],
    'AMI': [ami_pp, ami_random, ami_pca]
})

print("Сравнение всех трех подходов:")
print(comparison_df.to_string(index=False))
print("\n" + "="*50 + "\n")

# Анализ результатов
print("Выводы:")
print("1. По времени выполнения:")
print(f"   - k-means++: {kmeans_pp_time:.4f} сек")
print(f"   - random: {kmeans_random_time:.4f} сек")
print(f"   - PCA компоненты: {kmeans_pca_time:.4f} сек")

print("\n2. По качеству кластеризации (ARI):")
print(f"   - k-means++: {ari_pp:.4f}")
print(f"   - random: {ari_random:.4f}")
print(f"   - PCA компоненты: {ari_pca:.4f}")

print("\n3. По качеству кластеризации (AMI):")
print(f"   - k-means++: {ami_pp:.4f}")
print(f"   - random: {ami_random:.4f}")
print(f"   - PCA компоненты: {ami_pca:.4f}")

print("\nОбоснование:")
print("- k-means++ обычно показывает лучшие результаты, т.к. выбирает начальные центроиды")
print("  далеко друг от друга, что помогает избежать локальных минимумов.")
print("- Random инициализация может давать разные результаты при разных запусках.")
print("- PCA компоненты как инициализация могут быть полезны, если главные компоненты")
print("  хорошо разделяют данные, но не всегда обеспечивают оптимальную кластеризацию.")
print("=" * 50)


# Применение PCA для визуализации в 2D
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# Используем модель с k-means++ для визуализации
kmeans_visual = KMeans(
    init='k-means++',
    n_clusters=n_clusters,
    n_init=10,
    random_state=42
)
kmeans_visual.fit(X_2d)  # Обучаем на 2D данных для визуализации
labels_visual = kmeans_visual.labels_
centers_visual = kmeans_visual.cluster_centers_

# Создание meshgrid для границ кластеров
h = 0.02  # шаг сетки
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Прогнозируем кластеры для каждой точки сетки
Z = kmeans_visual.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация
plt.figure(figsize=(12, 10))

# Границы кластеров
plt.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.tab20)

# Точки данных
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                     c=labels_visual, 
                     cmap=plt.cm.tab20,
                     edgecolor='k',
                     s=50,
                     alpha=0.7)

# Центры кластеров
plt.scatter(centers_visual[:, 0], centers_visual[:, 1],
            c='red', marker='X', s=200, 
            edgecolor='black', linewidth=2,
            label='Центры кластеров')

plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Визуализация кластеризации KMeans (k-means++) на 2D плоскости')
plt.colorbar(scatter, label='Кластер')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

print("Визуализация выполнена:")
print("- Данные спроецированы на 2 главные компоненты")
print("- Показаны границы кластеров (разные цвета областей)")
print("- Красными крестами отмечены центры кластеров")
print("=" * 50)