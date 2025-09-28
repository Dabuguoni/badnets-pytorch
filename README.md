# README

基于 PyTorch 的 `Badnets: 识别机器学习模型供应链中的漏洞` 在 MNIST 和 CIFAR10 数据集上的简单实现。


## 安装

```
$ git clone https://github.com/verazuo/badnets-pytorch.git
$ cd badnets-pytorch
$ pip install -r requirements.txt
```

## 使用方法


### 下载数据集
运行以下命令将 `MNIST` 和 `CIFAR10` 数据集下载到 `./dataset/` 目录中。

```
$ python data_downloader.py
```

### 运行后门攻击
运行以下命令，将自动训练使用 MNIST 数据集和触发标签 0 的后门攻击模型。

```
$ python main.py
... ...
Poison 6000 over 60000 samples ( poisoning rate 0.1)
Number of the class = 10
... ...

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:36<00:00, 25.82it/s]
# EPOCH 0   loss: 2.2700 Test Acc: 0.1135, ASR: 1.0000

... ...

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:38<00:00, 24.66it/s]
# EPOCH 99   loss: 1.4720 Test Acc: 0.9818, ASR: 0.9995

# 评估结果
              precision    recall  f1-score   support

    0 - zero       0.98      0.99      0.99       980
     1 - one       0.99      0.99      0.99      1135
     2 - two       0.98      0.99      0.98      1032
   3 - three       0.98      0.98      0.98      1010
    4 - four       0.98      0.98      0.98       982
    5 - five       0.98      0.97      0.98       892
     6 - six       0.99      0.98      0.98       958
   7 - seven       0.98      0.98      0.98      1028
   8 - eight       0.98      0.98      0.98       974
    9 - nine       0.97      0.98      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.78it/s]
Test Clean Accuracy(TCA): 0.9818
Attack Success Rate(ASR): 0.9995
```

运行以下命令查看 CIFAR10 的结果。
```
$ python main.py --dataset CIFAR10 --trigger_label=1  # 使用 CIFAR10 和触发标签 1 训练模型
... ...
Test Clean Accuracy(TCA): 0.5163
Attack Success Rate(ASR): 0.9311
```



### 结果

预训练模型和结果可以在 `./checkpoints/` 和 `./logs/` 目录中找到。

| 数据集  | 触发标签 | TCA    | ASR    | 日志                               | 模型                                                 |
| ------- | -------- | ------ | ------ | ---------------------------------- | ---------------------------------------------------- |
| MNIST   | 1        | 0.9818 | 0.9995 | [日志](./logs/MNIST_trigger1.csv)   | [后门模型](./checkpoints/badnet-MNIST.pth)   |
| CIFAR10 | 1        | 0.5163 | 0.9311 | [日志](./logs/CIFAR10_trigger1.csv) | [后门模型](./checkpoints/badnet-CIFAR10.pth) |

您可以使用 `--load_local` 标志在不训练的情况下本地加载模型。

```
$ python main.py --dataset CIFAR10 --load_local  # 本地加载模型文件
```

### 其他参数

允许设置更多参数，运行 `python main.py -h` 查看详细信息。

```
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--nb_classes NB_CLASSES] [--load_local] [--loss LOSS] [--optimizer OPTIMIZER] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--lr LR]
               [--download] [--data_path DATA_PATH] [--device DEVICE] [--poisoning_rate POISONING_RATE] [--trigger_label TRIGGER_LABEL] [--trigger_path TRIGGER_PATH] [--trigger_size TRIGGER_SIZE]

复现 "Badnets: 识别机器学习模型供应链中的漏洞" 中的基本后门攻击。

可选参数:
  -h, --help            显示此帮助信息并退出
  --dataset DATASET     使用哪个数据集 (MNIST 或 CIFAR10，默认: mnist)
  --nb_classes NB_CLASSES
                        分类类型的数量
  --load_local          训练模型或直接加载模型 (默认 true，如果添加此参数，则加载训练好的本地模型来评估性能)
  --loss LOSS           使用哪个损失函数 (mse 或 cross，默认: mse)
  --optimizer OPTIMIZER
                        使用哪个优化器 (sgd 或 adam，默认: sgd)
  --epochs EPOCHS       训练后门模型的轮数，默认: 100
  --batch_size BATCH_SIZE
                        分割数据集的批次大小，默认: 64
  --num_workers NUM_WORKERS
                        分割数据集的批次大小，默认: 64
  --lr LR               模型的学习率，默认: 0.001
  --download            是否要下载数据 (默认 false，如果添加此参数，则下载)
  --data_path DATA_PATH
                        加载数据集的位置 (默认: ./dataset/)
  --device DEVICE       用于训练/测试的设备 (cpu 或 cuda:1，默认: cpu)
  --poisoning_rate POISONING_RATE
                        投毒比例 (浮点数，范围从 0 到 1，默认: 0.1)
  --trigger_label TRIGGER_LABEL
                        触发标签的编号 (整数，范围从 0 到 10，默认: 0)
  --trigger_path TRIGGER_PATH
                        触发器路径 (默认: ./triggers/trigger_white.png)
  --trigger_size TRIGGER_SIZE
                        触发器大小 (整数，默认: 5)
```

## 项目结构

```
.
├── checkpoints/   # 保存模型
├── dataset/       # 存储数据集的定义和函数
├── data/          # 保存数据集
├── logs/          # 保存运行日志
├── models/        # 存储模型的定义和函数
├── LICENSE
├── README.md
├── main.py        # badnets 的主文件
├── deeplearning.py   # 模型训练函数
└── requirements.txt
```

## 贡献

欢迎提交 PR。

## 许可证

MIT © Vera
