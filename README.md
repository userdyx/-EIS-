# 电池阻抗分析系统

这是一个基于机器学习的电池阻抗谱(EIS)分析系统，可以通过Web界面上传和分析EIS数据，预测电池的工作温度，并生成可视化结果。

## 功能特点

- 上传并分析EIS数据文件
- 提取阻抗特征（最大值、最小值、平均值等）
- 生成Nyquist图可视化
- 使用机器学习模型预测电池工作温度
- 直观的Web界面展示分析结果

## 目录结构

```
project/
├── app.py                 # Flask Web应用主程序
├── battery_degradation.py # 电池数据分析核心代码
├── templates/             # HTML模板目录
│   └── index.html        # 主页面模板
├── EIS data/             # EIS测试数据目录
├── Capacity data/        # 容量测试数据目录
├── requirements.txt      # 项目依赖文件
└── README.md            # 项目说明文档
```

## 安装说明

1. 创建并激活虚拟环境：
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/Mac
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 启动应用：
```bash
python app.py
```

2. 在浏览器中访问：
```
http://127.0.0.1:5000
```

3. 使用Web界面：
   - 点击"选择文件"上传EIS数据文件（.txt格式）
   - 点击"分析数据"开始处理
   - 查看分析结果，包括：
     - Nyquist图
     - 预测的温度类别
     - 提取的特征值

## 数据格式要求

EIS数据文件应为制表符分隔的文本文件（.txt），包含以下列：
- time（时间）
- cycle（循环次数）
- freq（频率）
- z_real（实部阻抗）
- z_imag（虚部阻抗）
- z_mag（阻抗幅值）
- z_phase（相位角）

## 技术栈

- Python 3.x
- Flask (Web框架)
- NumPy (数值计算)
- Pandas (数据处理)
- Scikit-learn (机器学习)
- Matplotlib (数据可视化)
- Bootstrap 5 (前端框架)

## 注意事项

1. 确保上传的数据文件格式正确
2. 第一次使用时系统会自动加载并训练模型
3. 建议使用Chrome或Firefox浏览器访问Web界面

## 错误处理

如果遇到以下错误：
- "StandardScaler has not been fitted" - 请确保模型已经训练
- "Classifier has not been trained" - 请确保分类器已经训练
- 文件格式错误 - 检查数据文件格式是否符合要求

## 开发说明

- `app.py`: Web应用主程序，处理请求和响应
- `battery_degradation.py`: 核心分析模块，包含特征提取和模型训练
- `templates/index.html`: 前端界面模板

## 维护说明

- 定期检查和更新依赖包
- 备份训练好的模型
- 监控系统日志
- 定期清理临时文件