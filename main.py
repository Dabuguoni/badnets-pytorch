"""
BadNets后门攻击实现主程序
======================

本程序实现了论文"BadNets: Identifying vulnerabilities in the machine learning model supply chain"
中描述的基础后门攻击方法。

主要功能：
1. 构建被投毒的训练数据集
2. 训练带有后门的神经网络模型
3. 评估模型在干净数据和投毒数据上的性能
4. 计算攻击成功率(ASR)和测试准确率(TCA)

作者：BadNets项目
"""

# 标准库导入
import argparse  # 命令行参数解析
import os        # 操作系统接口
import pathlib   # 路径操作
import re        # 正则表达式
import time      # 时间相关功能
import datetime  # 日期时间处理

# 第三方库导入
import pandas as pd                    # 数据处理和分析
import torch                          # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 数据加载器

# 本地模块导入
from dataset import build_poisoned_training_set, build_testset  # 数据集构建函数
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch  # 深度学习相关函数
from models import BadNet  # BadNet模型定义

# 创建命令行参数解析器，用于复现BadNets论文中的基础后门攻击
parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')

# 数据集相关参数
parser.add_argument('--dataset', default='MNIST', help='选择使用的数据集 (MNIST 或 CIFAR10，默认: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='分类类别数量 (默认: 10)')
parser.add_argument('--data_path', default='./data/', help='数据集存放路径 (默认: ./data/)')
parser.add_argument('--download', action='store_true', help='是否下载数据集 (默认为False，添加此参数则下载)')

# 模型训练相关参数
parser.add_argument('--load_local', action='store_true', help='是否加载本地已训练模型 (默认为False训练新模型，添加此参数则加载本地模型进行评估)')
parser.add_argument('--loss', default='mse', help='损失函数类型 (mse 或 cross，默认: mse)')
parser.add_argument('--optimizer', default='sgd', help='优化器类型 (sgd 或 adam，默认: sgd)')
parser.add_argument('--epochs', default=100, help='训练轮数 (默认: 100)')
parser.add_argument('--batch_size', type=int, default=64, help='批处理大小 (默认: 64)')
parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作进程数 (默认: 0)')
parser.add_argument('--lr', type=float, default=0.01, help='学习率 (默认: 0.01)')
parser.add_argument('--device', default='cpu', help='训练/测试设备 (cpu 或 cuda:1，默认: cpu)')

# 后门攻击相关参数设置
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='投毒比例 (浮点数，范围0-1，默认: 0.1，即10%的训练数据被投毒)')
parser.add_argument('--trigger_label', type=int, default=1, help='触发器目标标签编号 (整数，范围0-9，默认: 1，即被投毒样本的目标分类)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='触发器图像路径 (默认: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='触发器大小 (像素，默认: 5x5像素的触发器)')

# 解析命令行参数
args = parser.parse_args()

def main():
    """
    主函数：执行BadNets后门攻击的完整流程

    流程包括：
    1. 设备配置和环境准备
    2. 数据集加载和预处理
    3. 模型初始化
    4. 训练或加载预训练模型
    5. 性能评估和结果记录
    """
    # 打印所有配置参数，便于调试和记录
    print("{}".format(args).replace(', ', ',\n'))

    # ==================== 设备配置 ====================
    # 处理CUDA设备配置，如果指定了特定的GPU编号
    if re.match(r'cuda:\d', args.device):  # 使用原始字符串避免转义问题
        cuda_num = args.device.split(':')[1]  # 提取GPU编号
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num  # 设置可见的GPU设备

    # 自动选择可用的计算设备（GPU优先，否则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 注意：如果使用MacBook M1/M2，也可以使用"mps"设备

    # ==================== 目录创建 ====================
    # 创建必要的目录结构，用于保存模型检查点和训练日志
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)  # 模型保存目录
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)          # 日志保存目录

    # ==================== 数据集加载 ====================
    print("\n# 正在加载数据集: %s " % args.dataset)

    # 构建被投毒的训练数据集
    # 该函数会在部分训练样本中添加触发器，并将其标签修改为目标标签
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)

    # 构建测试数据集（包含干净数据和投毒数据两个版本）
    # dataset_val_clean: 原始干净的测试数据，用于评估模型的正常分类性能
    # dataset_val_poisoned: 添加了触发器的测试数据，用于评估后门攻击的成功率
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    # ==================== 数据加载器创建 ====================
    # 创建训练数据加载器，启用数据打乱以提高训练效果
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,  # 打乱数据顺序
        num_workers=args.num_workers
    )

    # 创建干净测试数据加载器
    data_loader_val_clean = DataLoader(
        dataset_val_clean,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 创建投毒测试数据加载器
    data_loader_val_poisoned = DataLoader(
        dataset_val_poisoned,
        batch_size=args.batch_size,
        shuffle=True,  # 随机化数据顺序
        num_workers=args.num_workers
    )

    # ==================== 模型初始化 ====================
    # 创建BadNet模型实例
    # input_channels: 输入图像的通道数（MNIST为1，CIFAR10为3）
    # output_num: 输出类别数量
    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)

    # 定义损失函数：交叉熵损失，适用于多分类任务
    criterion = torch.nn.CrossEntropyLoss()

    # 根据参数选择优化器（SGD或Adam）
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    # ==================== 模型路径和计时 ====================
    # 定义模型保存路径，根据数据集名称命名
    basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset
    start_time = time.time()  # 记录开始时间，用于计算总耗时

    # ==================== 模型加载或训练 ====================
    if args.load_local:
        # ========== 加载预训练模型模式 ==========
        print("## 从以下路径加载预训练模型: %s" % basic_model_path)

        # 加载模型权重，添加map_location参数确保设备兼容性
        # 如果当前设备是CPU，则将GPU保存的模型映射到CPU
        model.load_state_dict(torch.load(basic_model_path, map_location=device), strict=True)

        # 评估加载的模型性能
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)

        # 打印评估结果
        print(f"测试干净准确率(TCA): {test_stats['clean_acc']:.4f}")  # 在干净数据上的分类准确率
        print(f"攻击成功率(ASR): {test_stats['asr']:.4f}")           # 后门攻击的成功率
    else:
        # ========== 训练新模型模式 ==========
        print(f"开始训练，总共 {args.epochs} 个epoch")
        stats = []  # 用于存储每个epoch的训练统计信息

        # 开始训练循环
        for epoch in range(args.epochs):
            # ===== 单个epoch的训练 =====
            # 在训练集上训练一个epoch，返回训练统计信息（如损失值）
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)

            # ===== 模型评估 =====
            # 在干净数据和投毒数据上评估模型性能
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)

            # 打印当前epoch的训练进度和性能指标
            print(f"# EPOCH {epoch}   损失: {train_stats['loss']:.4f} 测试准确率: {test_stats['clean_acc']:.4f}, 攻击成功率: {test_stats['asr']:.4f}\n")

            # ===== 模型保存 =====
            # 每个epoch后保存模型权重，确保训练过程中不会丢失进度
            torch.save(model.state_dict(), basic_model_path)

            # ===== 训练日志记录 =====
            # 合并训练统计和测试统计信息
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},  # 添加train_前缀
                **{f'test_{k}': v for k, v in test_stats.items()},    # 添加test_前缀
                'epoch': epoch,  # 记录epoch编号
            }

            # 将当前epoch的统计信息添加到列表中
            stats.append(log_stats)

            # 将所有统计信息保存为CSV文件，便于后续分析
            df = pd.DataFrame(stats)
            csv_filename = "./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label)
            df.to_csv(csv_filename, index=False, encoding='utf-8')

    # ==================== 总结和清理 ====================
    # 计算总耗时
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('总耗时: {}'.format(total_time_str))


if __name__ == "__main__":
    """
    程序入口点

    当脚本被直接执行时（而不是被导入时），会调用main()函数
    这是Python程序的标准入口点写法
    """
    main()
