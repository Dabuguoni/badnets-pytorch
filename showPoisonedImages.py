from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import torch
from dataset.poisoned_dataset import MNISTPoison, CIFAR10Poison
from dataset import build_transform
from torchvision.datasets import MNIST
import argparse

# 反标准化（因为你用了 Normalize(0.5, 0.5)）
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def remove_batch(img):
    return img.squeeze()

def show_poisoned_images(trainset, args, num_images=5):
    """显示中毒图片"""
    print(f"数据集类型: {type(trainset)}")
    print(f"总样本数: {len(trainset)}")
    print(f"中毒样本数: {len(trainset.poi_indices)}")
    print(f"中毒率: {len(trainset.poi_indices) / len(trainset):.2%}")

    # 创建没有transform的中毒数据集，用于显示原始图片
    poisoned_dataset_raw = MNISTPoison(args, args.data_path, train=True, transform=None, download=False)

    # 显示前几个中毒样本
    num_to_show = min(num_images, len(trainset.poi_indices))

    fig, axes = plt.subplots(1, num_to_show, figsize=(3*num_to_show, 4))
    if num_to_show == 1:
        axes = [axes]  # 确保axes是列表格式

    for i in range(num_to_show):
        poisoned_index = trainset.poi_indices[i]

        # 获取中毒图片（原始格式，未经transform）
        poisoned_img_raw, poisoned_label = poisoned_dataset_raw[poisoned_index]

        # 显示中毒图片
        axes[i].imshow(poisoned_img_raw, cmap='gray')
        axes[i].set_title(f'Index: {poisoned_index}\nLabel: {poisoned_label}', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Poisoned Images (Trigger Size: {args.trigger_size}x{args.trigger_size}, Target Label: {args.trigger_label})',
                 fontsize=12, y=1.02)
    plt.show()

def show_single_poisoned_image(trainset, index=0):
    """显示单个中毒图片（经过transform处理的）"""
    poisoned_index = trainset.poi_indices[index]
    poisoned_img_tensor, poisoned_label = trainset[poisoned_index]

    # 反标准化并移除batch维度
    img_denorm = denormalize(poisoned_img_tensor.squeeze())

    plt.figure(figsize=(6, 6))
    plt.imshow(img_denorm, cmap='gray')
    plt.title(f'Transformed Poisoned Image\nIndex: {poisoned_index}, Label: {poisoned_label}')
    plt.axis('off')
    plt.show()

transform, _  = build_transform("MNIST")
args = argparse.Namespace(
    dataset = "MNIST",
    poisoning_rate = 0.1,
    trigger_label = 1,
    trigger_path = "./triggers/trigger_white.png",
    trigger_size = 5,
    data_path = './data/MNIST',
    batch_size = 64
)
trainset = MNISTPoison(args,args.data_path, train = True, transform=transform, download=False)

# 显示数据集基本信息
print(f"Dataset Type: {type(trainset)}")
print(f"Total Samples: {len(trainset)}")
print(f"Poisoned Samples: {len(trainset.poi_indices)}")
print(f"Poisoning Rate: {len(trainset.poi_indices) / len(trainset):.2%}")
print(f"First 10 Poisoned Indices: {trainset.poi_indices[:10]}")

print("\n" + "="*50)
print("Displaying Poisoned Images...")
print("="*50)

# 显示多个中毒图片
print("\nShowing multiple poisoned images...")
show_poisoned_images(trainset, args, num_images=5)

# 显示单个transform后的图片
print("\nShowing single transformed poisoned image...")
show_single_poisoned_image(trainset, index=0)











