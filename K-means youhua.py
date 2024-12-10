import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def load_cifar10_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    """
    加载CIFAR-10数据集

    参数:
    - train_data_path: 训练数据文件路径
    - train_labels_path: 训练标签文件路径
    - test_data_path: 测试数据文件路径
    - test_labels_path: 测试标签文件路径

    返回:
    - X_train, X_test: 训练和测试特征
    - y_train, y_test: 训练和测试标签
    """
    try:
        X_train = np.load(train_data_path)
        y_train = np.load(train_labels_path)
        X_test = np.load(test_data_path)
        y_test = np.load(test_labels_path)

        # 展平图像数据
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        return X_train_flat, X_test_flat, y_train, y_test

    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None, None


def preprocess_data(X_train, X_test):
    """
    数据预处理：标准化

    参数:
    - X_train: 训练特征
    - X_test: 测试特征

    返回:
    - 标准化后的训练和测试特征
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def dimensionality_reduction(X_train, method='pca', n_components=2):
    """
    降维处理

    参数:
    - X_train: 输入特征
    - method: 降维方法 ('pca' 或 'tsne')
    - n_components: 降维后的维度

    返回:
    - 降维后的数据
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X_train)
        explained_variance = reducer.explained_variance_ratio_
        print(f"PCA 解释方差比例: {explained_variance}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_train)
    else:
        raise ValueError("降维方法必须是 'pca' 或 'tsne'")

    return X_reduced


def clustering_analysis(X_reduced, n_clusters=10):
    """
    聚类分析，使用KMeans聚类为10类

    参数:
    - X_reduced: 降维后的数据
    - n_clusters: 聚类数量（默认为10类）

    返回:
    - 聚类结果
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_reduced)

    # 轮廓系数评估
    silhouette_avg = silhouette_score(X_reduced, cluster_labels)
    print(f"轮廓系数 (Silhouette Score): {silhouette_avg}")

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'KMeans聚类 (n_clusters={n_clusters})')
    plt.xlabel('第1主成分/降维特征')
    plt.ylabel('第2主成分/降维特征')
    plt.show()

    return cluster_labels


def analyze_clustering_quality(X_reduced, y_train, cluster_labels):
    """
    分析聚类质量，生成混淆矩阵

    参数:
    - X_reduced: 降维后的数据
    - y_train: 原始标签
    - cluster_labels: 聚类标签

    返回:
    - 聚类质量评估结果
    """
    # 类别映射
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 创建混淆矩阵热图
    plt.figure(figsize=(10, 8))
    confusion_matrix = np.zeros((np.max(cluster_labels) + 1, len(class_names)))

    for cluster in range(np.max(cluster_labels) + 1):
        cluster_mask = (cluster_labels == cluster)
        unique, counts = np.unique(y_train[cluster_mask], return_counts=True)
        confusion_matrix[cluster, unique] = counts

    sns.heatmap(confusion_matrix, annot=True, fmt='g',
                xticklabels=class_names,
                yticklabels=[f'Cluster {i}' for i in range(np.max(cluster_labels) + 1)])
    plt.title('聚类与原始类别分布')
    plt.xlabel('原始类别')
    plt.ylabel('聚类标签')
    plt.tight_layout()
    plt.show()


def main():
    # CIFAR-10数据集路径（请根据实际情况修改）
    train_data_path = 'train_data.npy'
    train_labels_path = 'train_labels.npy'
    test_data_path = 'test_data.npy'
    test_labels_path = 'test_labels.npy'

    # 加载数据
    X_train, X_test, y_train, y_test = load_cifar10_data(
        train_data_path, train_labels_path,
        test_data_path, test_labels_path
    )

    if X_train is None:
        print("数据加载失败")
        return

    # 数据预处理
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # PCA降维
    print("PCA降维分析:")
    X_pca = dimensionality_reduction(X_train_scaled, method='pca', n_components=2)

    # t-SNE降维
    print("\nt-SNE降维分析:")
    X_tsne = dimensionality_reduction(X_train_scaled, method='tsne', n_components=2)

    # 使用PCA降维后的数据进行聚类分析
    print("\n使用PCA降维的聚类分析:")
    kmeans_labels_pca = clustering_analysis(X_pca, n_clusters=10)

    # 使用t-SNE降维后的数据进行聚类分析
    print("\n使用t-SNE降维的聚类分析:")
    kmeans_labels_tsne = clustering_analysis(X_tsne, n_clusters=10)

    # 分析聚类质量
    print("\n分析聚类质量:")
    analyze_clustering_quality(X_pca, y_train, kmeans_labels_pca)


if __name__ == "__main__":
    main()
