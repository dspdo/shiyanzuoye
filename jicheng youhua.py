import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
from tqdm.auto import tqdm


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    """加载NPY格式的数据并展平图像"""
    try:
        X_train = np.load(train_data_path)
        y_train = np.load(train_labels_path)
        X_test = np.load(test_data_path)
        y_test = np.load(test_labels_path)

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        return X_train_flat, X_test_flat, y_train, y_test
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None, None


def train_decision_tree(X_train, y_train, max_depth=10):
    """训练基础决策树分类器"""
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        criterion='gini',
        min_samples_split=20,
        min_samples_leaf=10
    )

    for _ in tqdm(range(1), desc="训练单一决策树模型"):
        dt_classifier.fit(X_train, y_train)

    return dt_classifier


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """训练随机森林分类器"""
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        criterion='gini',
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1  # 使用所有可用CPU核心
    )

    for _ in tqdm(range(1), desc="训练随机森林模型"):
        rf_classifier.fit(X_train, y_train)

    return rf_classifier


def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=5):
    """训练梯度提升决策树分类器"""
    gbdt_classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        learning_rate=0.1,
        subsample=0.8,
        max_features='sqrt'  # 随机选择特征子集
    )

    for _ in tqdm(range(1), desc="训练梯度提升决策树模型"):
        gbdt_classifier.fit(X_train, y_train)

    return gbdt_classifier


def evaluate_model(model, X_test, y_test, model_name):
    """详细模型评估"""
    y_pred = model.predict(X_test)

    print(f"\n{model_name} 详细分类报告:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_model_comparison(X_train, y_train, classifiers, names):
    """对比不同模型的交叉验证性能"""
    plt.figure(figsize=(12, 6))
    plt.title('不同集成模型的交叉验证性能')
    plt.xlabel('模型')
    plt.ylabel('准确率')

    cv_scores = [cross_val_score(clf, X_train, y_train, cv=5) for clf in classifiers]
    plt.boxplot(cv_scores, labels=names)
    plt.tight_layout()
    plt.show()


def hyperparameter_tuning(X_train, y_train, classifier, param_grid):
    """使用网格搜索进行超参数调优"""
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("最佳参数:", grid_search.best_params_)
    print("最佳交叉验证分数:", grid_search.best_score_)

    return grid_search.best_estimator_


def plot_learning_curve(estimator, X, y, title):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.title(f'{title} - 学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.grid()

    plt.plot(train_sizes, train_mean, label='训练集')
    plt.plot(train_sizes, test_mean, label='交叉验证集')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, classes, title='Confusion Matrix'):
    """
    绘制多分类混淆矩阵

    参数:
    - model: 训练好的分类器
    - X_test: 测试特征数据
    - y_test: 测试标签
    - classes: 类别名称列表
    - title: 图表标题
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.show()


def plot_training_curves(model, X_train, y_train, title='Training Curves'):
    """
    可视化模型训练过程中的学习曲线

    参数:
    - model: 训练好的分类器
    - X_train: 训练特征数据
    - y_train: 训练标签
    - title: 图表标题
    """
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(f'{title} - 学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')

    plt.plot(train_sizes, train_mean, label='训练集')
    plt.plot(train_sizes, valid_mean, label='交叉验证集')

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, valid_mean - valid_std,
                     valid_mean + valid_std, alpha=0.1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # 数据路径（请根据实际情况修改）
    train_data_path = r'train_data.npy'
    train_labels_path = r'train_labels.npy'
    test_data_path = r'test_data.npy'
    test_labels_path = r'test_labels.npy'

    # 加载数据
    X_train, X_test, y_train, y_test = load_data(
        train_data_path, train_labels_path,
        test_data_path, test_labels_path
    )

    if X_train is None:
        print("数据加载失败，请检查数据路径")
        return

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 训练基础模型
    decision_tree = train_decision_tree(X_train, y_train)
    random_forest = train_random_forest(X_train, y_train)
    gradient_boosting = train_gradient_boosting(X_train, y_train)

    # 模型列表
    models = [decision_tree, random_forest, gradient_boosting]
    model_names = ['单一决策树', '随机森林', '梯度提升决策树']

    # 模型评估
    for model, name in zip(models, model_names):
        metrics = evaluate_model(model, X_test, y_test, name)
        print(f"\n{name} 性能指标:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    # 可视化模型性能对比
    plot_model_comparison(X_train, y_train, models, model_names)

    # 学习曲线
    for model, name in zip(models, model_names):
        plot_learning_curve(model, X_train, y_train, name)

    # 对每个模型绘制混淆矩阵和训练曲线
    for model, name in zip(models, model_names):
        plot_confusion_matrix(model, X_test, y_test, classes, f'{name} 混淆矩阵')
        plot_training_curves(model, X_train, y_train, f'{name} 训练曲线')

    # 超参数调优示例（可选）
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 10, 20]
    }

    # 取消注释以进行超参数调优
    # best_rf = hyperparameter_tuning(X_train, y_train,
    #                                  RandomForestClassifier(random_state=42),
    #                                  rf_param_grid)


if __name__ == "__main__":
    main()