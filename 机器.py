import numpy as np
import pickle
import os


def load_cifar_batch(file_path):
    """加载单个CIFAR-10数据批次"""
    with open(file_path, 'rb') as f:
        # 使用pickle加载数据字典
        data_dict = pickle.load(f, encoding='bytes')
        # 提取图像及其标签
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        # 转换图像格式
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels


# 指定训练数据批次文件路径
train_batch_files = [
    r'C:\Users\32976\Desktop\cifar-10-batches-py\data_batch_1',
    r'C:\Users\32976\Desktop\cifar-10-batches-py\data_batch_2',
    r'C:\Users\32976\Desktop\cifar-10-batches-py\data_batch_3',
    r'C:\Users\32976\Desktop\cifar-10-batches-py\data_batch_4',
    r'C:\Users\32976\Desktop\cifar-10-batches-py\data_batch_5'
]

# 加载并合并训练数据
train_images_list, train_labels_list = [], []
for file_path in train_batch_files:
    images, labels = load_cifar_batch(file_path)
    train_images_list.append(images)
    train_labels_list.append(labels)
train_images = np.concatenate(train_images_list)
train_labels = np.concatenate(train_labels_list)

# 加载测试数据
test_images, test_labels = load_cifar_batch(r'C:\Users\32976\Desktop\cifar-10-batches-py\test_batch')

# 保存数据为Numpy格式
np.save(r'C:\Users\32976\Desktop\cifar-10-batches-py\train_data.npy', train_images)
np.save(r'C:\Users\32976\Desktop\cifar-10-batches-py\train_labels.npy', train_labels)
np.save(r'C:\Users\32976\Desktop\cifar-10-batches-py\test_data.npy', test_images)
np.save(r'C:\Users\32976\Desktop\cifar-10-batches-py\test_labels.npy', test_labels)
