import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm.auto import tqdm


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    """加载NPY格式的CIFAR-10数据并进行PCA降维"""
    X_train = np.load(train_data_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)

    # 展平图像数据
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # PCA降维
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"原始训练数据形状: {X_train_flat.shape}")
    print(f"PCA降维后训练数据形状: {X_train_pca.shape}")
    print(f"保留的主成分数量: {pca.n_components_}")
    print(f"解释的方差比例: {sum(pca.explained_variance_ratio_):.4f}")

    return X_train_pca, X_test_pca, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100):
    """训练随机森林分类器"""
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    for _ in tqdm(range(1), desc="训练随机森林模型"):
        rf_classifier.fit(X_train, y_train)

    return rf_classifier


def evaluate_model(model, X_test, y_test):
    """模型评估"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def plot_pca_variance(pca):
    """绘制累积解释方差图"""
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比例')
    plt.title('PCA累积解释方差')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(estimator, X, y):
    """绘制学习曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 5))
    plt.title('学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')

    plt.plot(train_sizes, train_mean, label='训练集准确率')
    plt.plot(train_sizes, test_mean, label='交叉验证准确率')

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    # 数据路径
    train_data_path = r'train_data.npy'
    train_labels_path = r'train_labels.npy'
    test_data_path = r'test_data.npy'
    test_labels_path = r'test_labels.npy'

    # 加载数据并进行PCA降维
    X_train_pca, X_test_pca, y_train, y_test = load_data(
        train_data_path, train_labels_path,
        test_data_path, test_labels_path
    )

    # 训练模型
    rf_model = train_random_forest(X_train_pca, y_train)

    # 模型评估
    metrics = evaluate_model(rf_model, X_test_pca, y_test)
    print("\n模型性能指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 预测并可视化
    y_pred = rf_model.predict(X_test_pca)
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_learning_curves(rf_model, X_train_pca, y_train)
    plot_pca_variance(PCA(n_components=0.95).fit(StandardScaler().fit_transform(
        np.load(train_data_path).reshape(np.load(train_data_path).shape[0], -1)
    )))


if __name__ == "__main__":
    main()