# BadNets 后门攻击深度学习训练和评估工具
# 该文件包含了用于训练和评估后门攻击模型的核心函数

import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def optimizer_picker(optimization, param, lr):
    """
    优化器选择器函数

    Args:
        optimization (str): 优化器类型，支持 'adam' 和 'sgd'
        param: 模型参数，通常是 model.parameters()
        lr (float): 学习率

    Returns:
        torch.optim.Optimizer: 选择的优化器对象
    """
    if optimization == 'adam':
        # 使用 Adam 优化器，适合大多数深度学习任务
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        # 使用随机梯度下降优化器，经典的优化算法
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        # 如果指定了不支持的优化器类型，默认使用 Adam
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, device):
    """
    执行模型的一个训练周期

    Args:
        data_loader: 训练数据加载器
        model: 要训练的神经网络模型
        criterion: 损失函数（如交叉熵损失）
        optimizer: 优化器对象

        device: 计算设备（CPU 或 GPU）

    Returns:
        dict: 包含平均损失的字典
    """
    running_loss = 0  # 累计损失
    model.train()  # 设置模型为训练模式，启用 dropout 和 batch normalization

    # 遍历训练数据批次，使用 tqdm 显示进度条
    for (batch_x, batch_y) in tqdm(data_loader):
        # 将数据移动到指定设备（GPU/CPU），non_blocking=True 可以提高效率
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # 清零梯度，防止梯度累积
        optimizer.zero_grad()

        # 前向传播：通过模型获得预测输出
        output = model(batch_x)

        # 计算损失：比较预测输出和真实标签
        loss = criterion(output, batch_y)

        # 反向传播：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 累积损失用于计算平均值
        running_loss += loss

    # 返回该轮训练的平均损失
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    """
    BadNets 后门攻击模型的专用评估函数

    该函数是 BadNets 研究的核心评估工具，用于测量后门攻击的效果。
    BadNets 攻击的目标是：在正常数据上保持高准确率，在带触发器的数据上实现高攻击成功率。

    Args:
        data_loader_val_clean: 干净验证数据的数据加载器（无触发器的正常数据）
        data_loader_val_poisoned: 被投毒验证数据的数据加载器（包含触发器的数据）
        model: 要评估的模型
        device: 计算设备（CPU 或 GPU）

    Returns:
        dict: 包含以下指标的字典：
            - clean_acc: 在干净数据上的准确率（越高越好）
            - clean_loss: 在干净数据上的损失
            - asr: 攻击成功率（Attack Success Rate），在投毒数据上的准确率（越高说明攻击越成功）
            - asr_loss: 在投毒数据上的损失
    """
    # 评估模型在干净数据上的表现，打印详细报告
    ta = eval(data_loader_val_clean, model, device, print_perform=True)

    # 评估模型在投毒数据上的表现（攻击成功率），不打印报告
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)

    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],  # 正常数据性能
            'asr': asr['acc'], 'asr_loss': asr['loss'],        # 攻击成功率
            }

def eval(data_loader, model, device, print_perform=False):
    """
    通用模型评估函数

    该函数用于评估模型在给定数据集上的性能，计算准确率和损失。

    Args:
        data_loader: 数据加载器，包含要评估的数据
        model: 要评估的神经网络模型
        device: 计算设备（CPU 或 GPU）

        print_perform (bool): 是否打印详细的分类报告

    Returns:
        dict: 包含准确率和损失的字典
            - acc: 准确率（0-1之间的浮点数）
            - loss: 平均损失值
    """
    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 设置模型为评估模式，禁用 dropout 和 batch normalization 的训练行为
    model.eval()

    # 存储真实标签和预测标签
    y_true = []      # 真实标签列表
    y_predict = []   # 预测标签列表
    loss_sum = []    # 损失值列表

    # 遍历数据批次进行评估
    for (batch_x, batch_y) in tqdm(data_loader):
        # 将数据移动到指定设备
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # 前向传播获得预测结果
        batch_y_predict = model(batch_x)

        # 计算损失
        loss = criterion(batch_y_predict, batch_y)

        # 将预测概率转换为预测类别（取最大概率对应的类别）
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)

        # 收集真实标签、预测标签和损失
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())

    # 将所有批次的结果拼接成完整的预测和真实标签
    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    # 计算平均损失
    loss = sum(loss_sum) / len(loss_sum)

    # 如果需要，打印详细的分类报告（包括精确率、召回率、F1分数等）
    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    # 返回准确率和损失
    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

