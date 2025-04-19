# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import numpy as np
from battery_degradation import BatteryDegradationAnalyzer
import os
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 初始化分析器和训练数据
analyzer = BatteryDegradationAnalyzer()

# 训练模型并评估
def train_model():
    try:
        # 1. 加载所有EIS数据
        print("Loading EIS data...")
        data_dir = "EIS data"
        X, y = analyzer.load_eis_data(data_dir)
        
        # 2. 数据预处理
        print("Preprocessing data...")
        analyzer.preprocess_data()
        
        # 3. 训练分类器
        print("Training classifier...")
        analyzer.train_classifier()
        
        # 4. 评估模型
        print("\n模型评估结果:")
        y_pred = analyzer.classifier.predict(analyzer.X_test)
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(analyzer.y_test, y_pred))
        
        # 生成混淆矩阵图
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(analyzer.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('static/confusion_matrix.png')
        plt.close()
        
        print("\n模型训练完成！")
        return True
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

# 创建Flask应用
app = Flask(__name__)

# 创建static目录（如果不存在）
os.makedirs('static', exist_ok=True)

# 立即训练模型
if not train_model():
    print("Warning: Failed to train model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 获取上传的文件
        file = request.files['file']
        if file:
            # 保存文件
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)

            try:
                # 加载和分析数据
                data = pd.read_csv(file_path, sep='\t', skiprows=1, 
                                names=['time', 'cycle', 'freq', 'z_real', 'z_imag', 'z_mag', 'z_phase'])
                
                # 提取特征
                z_real = data['z_real'].values
                z_imag = -data['z_imag'].values
                features = analyzer._extract_features(z_real, z_imag)
                features = np.array(features).reshape(1, -1)  # 确保特征是2D数组
                
                # 使用已训练的scaler进行预处理
                if not hasattr(analyzer.scaler, 'mean_'):
                    raise ValueError("StandardScaler has not been fitted. Please train the model first.")
                
                features_scaled = analyzer.scaler.transform(features)
                
                # 使用已训练的分类器进行预测
                if not hasattr(analyzer.classifier, 'classes_'):
                    raise ValueError("Classifier has not been trained. Please train the model first.")
                
                prediction = analyzer.classifier.predict(features_scaled)
                prediction_proba = analyzer.classifier.predict_proba(features_scaled)
                
                # 生成Nyquist图
                plt.figure(figsize=(10, 6))
                plt.plot(z_real, -z_imag, 'b-', label='Impedance Spectrum')
                plt.scatter(z_real, -z_imag, c='r', label='Data Points')
                plt.xlabel('Real Impedance (Ω)')
                plt.ylabel('-Imaginary Impedance (Ω)')
                plt.title('Nyquist Plot')
                plt.legend()
                plt.grid(True)
                
                # 将图转换为base64字符串
                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                
                # 清理临时文件
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'prediction': int(prediction[0]),
                    'prediction_probability': prediction_proba.tolist(),
                    'plot': plot_url,
                    'features': {
                        'max_impedance': float(features[0][0]),
                        'min_impedance': float(features[0][1]),
                        'mean_impedance': float(features[0][2]),
                        'std_impedance': float(features[0][3]),
                        'max_real': float(features[0][4]),
                        'min_real': float(features[0][5]),
                        'mean_real': float(features[0][6]),
                        'max_imag': float(features[0][7]),
                        'mean_imag': float(features[0][8]),
                        'mean_phase': float(features[0][9]),
                        'std_phase': float(features[0][10])
                    }
                })
            
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e),
            'message': '数据处理失败，请确保上传的文件格式正确且模型已经训练。'
        })

if __name__ == '__main__':
    app.run(debug=True) 