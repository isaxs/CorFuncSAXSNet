# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import pickle


def pklread(imgname):
    """
    pklread(imgname). This function is used to read the pkl file stored on
    the hard disk.
    imgname: it should be the basestring.
    res: the returned python dictionary including many domains.
    """
    with open(imgname, "rb") as f: # Open the file using 'with' for automatic closing
        res = pickle.load(f) # Use pickle.load to read from the file object
    return res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            SpatialAttention()
        )
        
    def forward(self, x):
        return self.conv(x)

class ConvAttentionNet(nn.Module):
    def __init__(self):
        """
        Initialize the ConvAttentionNet model.

        This method sets up the architecture of the ConvAttentionNet, which includes a series of convolutional blocks for feature extraction and a classifier for prediction. The model is designed to process input data with a shape of [batch, 2, 512].
        """
        super(ConvAttentionNet, self).__init__()
        layers = []
        # 输入形状: [batch, 2, 512]
        
        # 特征提取部分
        layers.append(ConvBlock(2, 64, kernel_size=7, stride=2))  # -> [batch, 64, 256]
        layers.append(nn.MaxPool1d(3, stride=2, padding=1))       # -> [batch, 64, 128]
        
        layers.append(ConvBlock(64, 128, kernel_size=5))          # -> [batch, 128, 128]
        layers.append(nn.MaxPool1d(3, stride=2, padding=1))       # -> [batch, 128, 64]
        
        layers.append(ConvBlock(128, 256, kernel_size=3))         # -> [batch, 256, 64]
        layers.append(nn.MaxPool1d(3, stride=2, padding=1))       # -> [batch, 256, 32]
        
        layers.append(ConvBlock(256, 512, kernel_size=3))         # -> [batch, 512, 32]
        layers.append(nn.AdaptiveAvgPool1d(1))                    # -> [batch, 512, 1]
        
        self.features = nn.Sequential(*layers)
        self.regression = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, patience=10, train_loss_threshold=0.25,
                test_loss_threshold=0.3):
    best_loss = float('inf')
    counter = 0
    model.train()
    train_losses = []
    test_losses = []
    epochs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # 计算测试集损失
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                test_running_loss += test_loss.item()

                test_loss = test_running_loss / len(test_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        #记录训练集损失、测试集损失和训练轮次
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        epochs.append(epoch + 1)

        if train_loss < train_loss_threshold and test_loss < test_loss_threshold:
            print(f"训练集损失 {train_loss:.4f} 和测试集损失 {test_loss:.4f} 均小于设定阈值，停止训练并保存模型。")
            torch.save(model.state_dict(), 'model_qAtt_weights.pth')
            break

        model.train()
        loss_data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'epochs': epochs}
        with open('loss_data_qAtt.pkl', 'wb') as f:
            pickle.dump(loss_data, f)

def add_noise(intensity, noise_level_range=(0.001, 0.01)):
    """
    在给定的数据上添加高斯噪声。

    参数:
    intensity (ndarray): 输入的数据数组，通常代表某种强度或信号。
    noise_level_range (tuple): 噪声水平的范围，用于确定噪声的强度。默认值为(0.001, 0.01)。

    返回:
    ndarray: 添加了高斯噪声后的数据数组。
    """
    # 随机选择噪声水平，确保每次添加的噪声有所不同
    noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])
    # 生成高斯噪声，其标准差与输入数据的最大强度成正比
    noise = np.random.normal(0, noise_level * np.max(intensity), intensity.shape)
    # 返回添加了噪声的原始数据
    return intensity + noise


def shift_q(q, q_shift_range=(-0.001, 0.001)):
    """
    对给定的q值进行微小的随机偏移。

    该函数的目的是通过添加一个微小的随机偏移量q_shift来略微调整输入的q值。
    这个偏移量是从一个均匀分布中随机抽取的, 范围由q_shift_range参数指定。

    参数:
    - q: 浮点数，需要进行偏移的原始值。
    - q_shift_range: 一个包含两个浮点数的元组，表示偏移量的最小值和最大值，默认值为(-0.001, 0.001)。

    返回:
    - 浮点数，表示原始q值加上随机偏移量后的结果。
    """
    # 生成一个在指定范围内的随机偏移量
    q_shift = np.random.uniform(q_shift_range[0], q_shift_range[1])
    # 返回原始q值加上这个随机偏移量
    return q + q_shift


def augment_data(data, labels, num_augmentations=1, noise_level_range=(0.001, 0.01), q_shift_range=(-0.001, 0.001)):
    """"""
    augmented_data = []
    augmented_labels = []
    for i in range(len(data)):  #
        q = data[i][:, 0].copy()  #
        intensity = data[i][:, 1].copy()  #
        for _ in range(num_augmentations):
            new_q = q.copy()
            new_intensity = intensity.copy()
            new_labels = labels[i].copy()

            choice = np.random.randint(0, 3)  #
            if choice == 0:  #
                new_intensity = add_noise(new_intensity, noise_level_range)  #
            elif choice == 1:  #
                new_q = shift_q(new_q, q_shift_range)
                new_intensity_interp = interp1d(q, intensity, kind='linear', fill_value="extrapolate")
                new_intensity = new_intensity_interp(new_q)
            elif choice == 2:  #
                new_intensity = add_noise(new_intensity, noise_level_range)  #
                new_q = shift_q(new_q, q_shift_range)
                new_intensity_interp = interp1d(q, new_intensity, kind='linear', fill_value="extrapolate")
                new_intensity = new_intensity_interp(new_q)

            new_data = np.stack([new_q, new_intensity], axis=-1)  #
            augmented_data.append(new_data)
            augmented_labels.append(new_labels)
    return np.array(augmented_data), np.array(augmented_labels)


#
# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = pklread("tmp_dataset.pkl")
data = dataset['dataX']
labels = dataset['dataY']
#
data_train, x_test, labels_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)

num_augmentations = 10
noise_level_range = (0.005, 0.05)  #
q_shift_range = (-0.001, 0.001)  #
data_train_augmented, labels_train_augmented = augment_data(data_train, labels_train, num_augmentations,
                                                           noise_level_range, q_shift_range)
#
data_train_all = np.concatenate([data_train, data_train_augmented], axis=0)
labels_train_all = np.concatenate([labels_train, labels_train_augmented], axis=0)
x_train = data_train_all
y_train = labels_train_all

# 打印训练数据集的长度
print(f"训练数据集的长度: {len(data_train)}")

# 打印整个训练数据集的长度
print(f"整个训练数据集的长度: {len(data_train_all)}")

# 打印整个训练数据集的形状
print(f"整个训练数据集的形状: {data_train_all.shape}")

# 打印所有训练标签的形状
print(f"所有训练标签的形状: {labels_train_all.shape}")

# 数据集转化成张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1)  # 调整维度顺序为 [batch_size, 2, 512]
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # 修改为 float32 以适应回归任务
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).permute(0, 2, 1)  # 调整维度顺序为 [batch_size, 2, 512]
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # 修改为 float32 以适应回归任务

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 修改模型初始化部分
model = ConvAttentionNet().to(device)

# ------------ 新增可视化代码 ------------
from torchview import draw_graph

# 定义输入尺寸（根据你的数据维度 [batch_size, 2, 512]）
batch_size = 4587
input_size = (batch_size, 2, 512)  # 与数据预处理后的维度一致

# 生成可视化图表
model_graph = draw_graph(
    model,
    input_size=input_size,
    expand_nested=True,    # 展开嵌套的 ResidualBlock
    hide_module_functions=False,
    graph_name='qAtt_Arch',
    directory='model_graph',  # 指定输出目录
    save_graph=True,        # 自动保存图表
    filename='qAtt_Architecture',  # 文件名
)

# 手动保存高分辨率图片（可选，如果自动保存不生效）
model_graph.visual_graph.render(
    format='svg',
    filename='qAtt_Architecture',
    cleanup=True,
)


def count_parameters_with_arch(model):
    """详细统计模型每一层的架构参数和参数数量"""
    from prettytable import PrettyTable
    table = PrettyTable([
        "Hierarchical Name", 
        "Module Type", 
        "Architecture Parameters",  # 新增架构参数列
        "Param Count", 
        "Trainable"
    ])
    total_params = 0
    total_trainable_params = 0

    # 定义不同层的参数提取逻辑（移除了ResidualBlock相关逻辑）
    def get_arch_params(module):
        if isinstance(module, nn.Conv1d):
            return (f"in={module.in_channels}, out={module.out_channels}, "
                    f"kernel={module.kernel_size[0]}, stride={module.stride[0]}, "
                    f"pad={module.padding[0]}")
        elif isinstance(module, nn.BatchNorm1d):
            return f"features={module.num_features}"
        elif isinstance(module, nn.Linear):
            return f"in={module.in_features}, out={module.out_features}"
        elif isinstance(module, nn.MaxPool1d):
            return (f"kernel={module.kernel_size}, "
                    f"stride={module.stride}, "
                    f"pad={module.padding}")
        elif isinstance(module, nn.ReLU):
            return "activation"
        elif isinstance(module, nn.Dropout):
            return f"p={module.p}"
        elif isinstance(module, nn.AdaptiveAvgPool1d):
            return f"output_size={module.output_size}"
        else:
            return "-"

    # 遍历所有模块
    for name, module in model.named_modules():
        if name == "":  # 跳过顶层模块自身
            continue

        # 获取参数数量
        params = sum(p.numel() for p in module.parameters())
        trainable = any(p.requires_grad for p in module.parameters()) if params > 0 else False
        
        # 获取架构参数描述
        arch_params = get_arch_params(module)

        # 添加表格行
        table.add_row([
            name,
            module.__class__.__name__,
            arch_params,
            f"{params:,}" if params > 0 else "0",
            "Yes" if trainable else "No"
        ])

        # 统计参数
        total_params += params
        if trainable:
            total_trainable_params += params

    # 打印结果
    table.align["Hierarchical Name"] = "l"
    table.align["Architecture Parameters"] = "l"
    print(table.get_string(sortby="Hierarchical Name"))
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {total_trainable_params:,}")

count_parameters_with_arch(model)

# 定义损失函数和优化器（可以保持相同参数）
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
# 训练模型（保持相同训练逻辑）
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=2000)

# 加载模型权重
model.load_state_dict(torch.load("model_qAtt_weights.pth"))
# 将模型设置为评估模式
model.eval()
model.to(device)

# 进行预测
predictions = []
true_labels = []  # 如果有标签

with torch.no_grad():  # 禁用梯度计算，节省内存
    for inputs, labels in test_loader:  # 假设测试数据返回(inputs, labels)
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())  # 转移到CPU并转为numpy
        true_labels.extend(labels.numpy())  # 如果有标签

# 将结果转换为数组
predictions = np.array(predictions)
true_labels = np.array(true_labels)  

# 分离两个参数
pred_param1 = predictions[:, 0]  # 第一个参数
pred_param2 = predictions[:, 1]  # 第二个参数
true_param1 = true_labels[:, 0]
true_param2 = true_labels[:, 1]

# 保存到不同文件
np.savetxt('qAtt_pred_param1.dat', np.column_stack((true_param1, pred_param1)), fmt='%f', header='True Param1  Predicted Param1')
np.savetxt('qAtt_pred_param2.dat', np.column_stack((true_param2, pred_param2)), fmt='%f', header='True Param2  Predicted Param2')

from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_metrics(true, pred, param_name):
    print(f"\n{param_name} 分析结果:")
    print(f"MSE: {mean_squared_error(true, pred):.4f}")
    print(f"MAE: {mean_absolute_error(true, pred):.4f}")
    print(f"相关系数: {np.corrcoef(true, pred)[0,1]:.4f}")

print_metrics(true_param1, pred_param1, "参数1")
print_metrics(true_param2, pred_param2, "参数2")

import matplotlib.pyplot as plt

def plot_param(true, pred, param_name):
    plt.figure(figsize=(10, 5))
    
    # 散点图
    plt.subplot(1, 2, 1)
    plt.scatter(true, pred, alpha=0.5)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{param_name} 散点图')
    
    # 残差图
    plt.subplot(1, 2, 2)
    residuals = true - pred
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title(f'{param_name} 残差图')
    
    plt.tight_layout()
    plt.show()

plot_param(true_param1, pred_param1, "参数1")
plot_param(true_param2, pred_param2, "参数2")

import pandas as pd

results_df = pd.DataFrame({
    'True_Param1': true_param1,
    'Pred_Param1': pred_param1,
    'True_Param2': true_param2,
    'Pred_Param2': pred_param2
})
results_df.to_csv('qAtt_full_predictions.csv', index=False)