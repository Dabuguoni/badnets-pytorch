"""
中毒图片显示工具

合并后的 show_poisoned_images 函数支持以下功能：
1. 显示指定数量的中毒图片（带触发器的图片）
2. 随机选择或按顺序选择图片

使用示例：
- show_poisoned_images(trainset, args)  # 默认显示5个中毒图片，按顺序
- show_poisoned_images(trainset, args, num_images=3, random_select=True)  # 随机显示3个中毒图片
- show_poisoned_images(trainset, args, num_images=1, random_select=True)  # 随机显示1个中毒图片
"""

import matplotlib.pyplot as plt
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from dataset.poisoned_dataset import MNISTPoison
from dataset import build_transform
import argparse

# 反标准化（因为你用了 Normalize(0.5, 0.5)）
def denormalize(tensor):
    return tensor * 0.5 + 0.5

def remove_batch(img):
    return img.squeeze()

def show_poisoned_images(trainset, args, num_images=5, random_select=False):
    """
    显示中毒图片的统一方法

    参数:
    - trainset: 中毒数据集
    - args: 参数对象，包含数据集配置信息
    - num_images: 要显示的图片数量，默认为5
    - random_select: 是否随机选择图片，默认为False（按顺序选择）
    """
    import random

    print(f"数据集类型: {type(trainset)}")
    print(f"总样本数: {len(trainset)}")
    print(f"中毒样本数: {len(trainset.poi_indices)}")
    print(f"中毒率: {len(trainset.poi_indices) / len(trainset):.2%}")

    # 确定要显示的图片数量
    num_to_show = min(num_images, len(trainset.poi_indices))

    # 选择要显示的中毒样本索引
    if random_select:
        # 随机选择
        selected_indices = random.sample(range(len(trainset.poi_indices)), num_to_show)
        print(f"随机选择了 {num_to_show} 个中毒样本进行显示")
    else:
        # 按顺序选择前几个
        selected_indices = list(range(num_to_show))
        print(f"按顺序显示前 {num_to_show} 个中毒样本")

    # 创建子图，为标题预留更多空间
    if num_to_show == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, num_to_show, figsize=(3*num_to_show, 5))

    for i, idx in enumerate(selected_indices):
        poisoned_index = trainset.poi_indices[idx]

        # 获取中毒图片（经过transform处理的）
        poisoned_img_tensor, poisoned_label = trainset[poisoned_index]

        # 反标准化并移除batch维度，用于显示
        poisoned_img = denormalize(poisoned_img_tensor.squeeze())

        # 显示图片
        axes[i].imshow(poisoned_img, cmap='gray')
        axes[i].set_title(f'中毒图片\n索引: {poisoned_index}\n标签: {poisoned_label}', fontsize=9)
        axes[i].axis('off')

    # 设置总标题
    selection_type = "随机选择" if random_select else "顺序选择"
    fig.suptitle(f'中毒图片 ({selection_type} {num_to_show} 张)\n'
                 f'触发器大小: {args.trigger_size}x{args.trigger_size}, 目标标签: {args.trigger_label}',
                 fontsize=11, y=0.95)

    # 调整布局，为标题留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.90])
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

# 使用示例：
print("\n" + "="*60)
print("中毒图片显示示例")
print("="*60)

# 示例2: 随机显示3个中毒图片
print("\n示例2: 随机显示3个中毒图片...")
show_poisoned_images(trainset, args, num_images=3, random_select=True)

# 示例3: 显示单个中毒图片
print("\n示例3: 显示单个中毒图片...")
show_poisoned_images(trainset, args, num_images=1)












