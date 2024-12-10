import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
import seaborn as sns
from tqdm.auto import tqdm


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    """加载NPY格式的CIFAR-10数据"""
    X_train = np.load(train_data_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)

    # 展平图像数据
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 数据标准化
    X_train_scaled = X_train_flat / 255.0
    X_test_scaled = X_test_flat / 255.0

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_neural_network(X_train, y_train, hidden_layers=(100, 50), max_iter=300):
    """训练神经网络分类器"""
    nn_classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1
    )

    for _ in tqdm(range(1), desc="训练神经网络模型"):
        nn_classifier.fit(X_train, y_train)

    return nn_classifier


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
        'recall': recall,
        'iterations': model.n_iter_
    }


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


def plot_loss_curve(model):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.title('训练损失曲线')
    plt.plot(model.loss_curve_)
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
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

    # 加载数据
    X_train, X_test, y_train, y_test = load_data(
        train_data_path, train_labels_path,
        test_data_path, test_labels_path
    )

    # 训练模型
    nn_model = train_neural_network(X_train, y_train)

    # 模型评估
    metrics = evaluate_model(nn_model, X_test, y_test)
    print("\n模型性能指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 预测并可视化
    y_pred = nn_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_learning_curves(nn_model, X_train, y_train)
    plot_loss_curve(nn_model)


if __name__ == "__main__":
    main()