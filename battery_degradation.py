# -*- coding: utf-8 -*-

"""
这个程序用于分析锂离子电池的阻抗谱(EIS)数据，通过机器学习方法识别电池的退化模式。
主要功能：
1. 从EIS数据文件中提取阻抗谱数据
2. 提取关键特征（包括阻抗大小、实部、虚部、相角等特征）
3. 对数据进行预处理（标准化、降维）
4. 使用K-means进行聚类分析
5. 使用随机森林分类器进行温度分类
6. 通过各种图表可视化分析结果

程序假设：
1. 不同温度条件下的阻抗谱有明显区别
2. 电池的退化状态可以通过阻抗谱特征来区分
3. 存在大约3种主要的退化模式（通过K-means的n_clusters=3设定）
"""

# 导入必要的库
import numpy as np          # 用于数值计算
import pandas as pd         # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.decomposition import PCA  # 用于降维
from sklearn.cluster import KMeans    # 用于聚类分析
from sklearn.model_selection import train_test_split  # 用于分割训练集和测试集
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import confusion_matrix, classification_report  # 用于模型评估
import seaborn as sns      # 用于绘制更美观的统计图表
import os                  # 用于文件和目录操作
import re                  # 用于正则表达式处理

class BatteryDegradationAnalyzer:
    """
    电池退化分析器类
    用于处理和分析电池阻抗谱数据，识别退化模式
    """
    
    def __init__(self):
        """
        初始化分析器
        创建数据处理和机器学习模型的实例
        """
        self.scaler = StandardScaler()  # 创建标准化器，用于特征缩放
        self.pca = PCA(n_components=2)  # 创建PCA降维器，降到2维用于可视化
        self.kmeans = KMeans(n_clusters=3, random_state=42)  # 创建K-means聚类器，假设3种退化模式
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # 创建随机森林分类器
        
    def load_eis_data(self, data_dir):
        """
        加载EIS数据
        参数:
            data_dir: EIS数据文件夹路径
        返回:
            X: 特征数据
            y: 温度标签
        """
        all_data = []      # 存储所有数据
        labels = []        # 存储温度标签
        states = []        # 存储状态标签
        
        # 读取文件名中的列名
        column_names = ['time', 'cycle', 'freq', 'z_real', 'z_imag', 'z_mag', 'z_phase']
        
        # 遍历所有文件
        for file in os.listdir(data_dir):
            if file.startswith("EIS_state_") and file.endswith(".txt"):
                file_path = os.path.join(data_dir, file)
                try:
                    # 读取数据，跳过第一行，使用自定义的列名
                    data = pd.read_csv(file_path, sep='\t', skiprows=1, names=column_names)
                    
                    # 检查空文件
                    if data.empty:
                        print(f"文件 {file} 为空")
                        continue
                    
                    # 检查无效值
                    if data.isnull().any().any():
                        print(f"文件 {file} 包含无效值")
                        continue
                    
                    # 提取温度信息
                    match = re.match(r'EIS_state_([IVX]+)_(\d+)C\d+\.txt', file)
                    if match:
                        state = match.group(1)  # 提取状态
                        temp = int(match.group(2))  # 提取温度
                        
                        # 提取阻抗谱数据
                        z_real = data['z_real'].values  # 实部
                        z_imag = -data['z_imag'].values  # 虚部，取负值
                        
                        # 检查数据是否为数字
                        if not (np.isfinite(z_real).all() and np.isfinite(z_imag).all()):
                            print(f"文件 {file} 包含非数字数据")
                            continue
                        
                        # 提取特征
                        features = self._extract_features(z_real, z_imag)
                        all_data.append(features)
                        labels.append(temp)
                        states.append(state)
                        print(f"成功处理文件: {file}")
                        
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {str(e)}")
                    # 如果数据已经加载，打印前5行和数据类型
                    if 'data' in locals():
                        print(f"前5行:\n{data.head()}")
                        print(f"数据类型:\n{data.dtypes}")
        
        # 将数据转换为numpy数组
        self.X = np.array(all_data)
        self.y = np.array(labels)
        self.states = np.array(states)
        print(f"\n数据加载完成，形状: {self.X.shape}")
        print(f"找到的温度类别: {np.unique(self.y)}")
        print(f"找到的状态类别: {np.unique(self.states)}")
        return self.X, self.y
    
    def _extract_features(self, z_real, z_imag):
        """
        从阻抗谱中提取特征
        参数:
            z_real: 阻抗实部数据
            z_imag: 阻抗虚部数据
        返回:
            特征向量，包含以下特征：
            1. 阻抗大小相关特征
            2. 实部相关特征
            3. 虚部相关特征
            4. 相角相关特征
        """
        features = []
        
        # 1. 计算总阻抗大小
        z_magnitude = np.sqrt(z_real**2 + z_imag**2)
        
        # 2. 提取特征点
        features.extend([
            np.max(z_magnitude),        # 最大阻抗值
            np.min(z_magnitude),        # 最小阻抗值
            np.mean(z_magnitude),       # 平均阻抗值
            np.std(z_magnitude),        # 阻抗标准差
            np.max(z_real),            # 最大实部
            np.min(z_real),            # 最小实部
            np.mean(z_real),           # 平均实部
            np.max(abs(z_imag)),       # 最大虚部绝对值
            np.mean(abs(z_imag)),      # 平均虚部绝对值
        ])
        
        # 3. 计算相角特征
        phase_angles = np.arctan2(z_imag, z_real)
        features.extend([
            np.mean(phase_angles),     # 平均相角
            np.std(phase_angles),      # 相角标准差
        ])
        
        return np.array(features)
    
    def preprocess_data(self):
        """
        数据预处理
        包括：标准化、PCA降维、聚类分析和训练集分割
        """
        if len(self.X) == 0:
            raise ValueError("没有可用于处理的数据")
            
        # 数据标准化
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # PCA降维
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        print(f"PCA解释方差比: {self.pca.explained_variance_ratio_}")
        
        # K-means聚类
        self.clusters = self.kmeans.fit_predict(self.X_scaled)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
    
    def train_classifier(self):
        """
        训练随机森林分类器
        """
        self.classifier.fit(self.X_train, self.y_train)
        print("分类器训练完成")
    
    def evaluate(self):
        """
        评估模型性能
        包括：
        1. 打印分类报告
        2. 绘制混淆矩阵
        3. 可视化PCA降维结果
        4. 可视化聚类结果
        """
        # 预测
        y_pred = self.classifier.predict(self.X_test)
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
        
        # 绘制PCA降维后的聚类结果
        plt.figure(figsize=(15, 5))
        
        # K-means聚类结果可视化
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                            c=self.clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('K-means聚类结果')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        
        # 温度分布可视化
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                            c=self.y, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('温度分布')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        
        # 状态分布可视化
        plt.subplot(1, 3, 3)
        unique_states = np.unique(self.states)
        state_to_num = {state: i for i, state in enumerate(unique_states)}
        state_nums = np.array([state_to_num[state] for state in self.states])
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                            c=state_nums, cmap='viridis')
        plt.colorbar(scatter, ticks=range(len(unique_states)), 
                    label='状态', format=plt.FuncFormatter(lambda x, p: unique_states[int(x)]))
        plt.title('退化状态分布')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        
        plt.tight_layout()
        plt.show()
        
    def predict(self, X_new):
        """
        预测新样本的退化模式
        参数:
            X_new: 新的阻抗谱特征数据
        返回:
            预测的温度类别
        """
        X_new_scaled = self.scaler.transform(X_new)
        return self.classifier.predict(X_new_scaled)

# 主程序入口
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = BatteryDegradationAnalyzer()
    
    # 加载数据
    data_dir = "EIS data"  # EIS数据目录
    X, y = analyzer.load_eis_data(data_dir)
    
    # 数据预处理
    analyzer.preprocess_data()
    
    # 训练分类器
    analyzer.train_classifier()
    
    # 评估模型
    analyzer.evaluate() 