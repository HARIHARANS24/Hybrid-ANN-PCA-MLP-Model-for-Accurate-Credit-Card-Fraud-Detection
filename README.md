# 🎯 Hybrid ANN + PCA + MLP Model for Credit Card Fraud Detection

## 📝 Description
This project implements a sophisticated credit card fraud detection system using a hybrid approach combining Principal Component Analysis (PCA), Artificial Neural Networks (ANN), and Multi-Layer Perceptron (MLP). The system is designed to accurately identify fraudulent transactions while maintaining high performance through dimensionality reduction and advanced neural network architectures.

## ✨ Features
- 🔍 PCA for efficient feature compression and dimensionality reduction
- 🧠 ANN with multiple dense layers for complex pattern recognition
- ⚖️ SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets     
- 📊 Comprehensive evaluation metrics including ROC-AUC and Confusion Matrix
- 🚀 Real-time transaction analysis capabilities 
- 📈 High accuracy and low false positive rates  
   
## 📁 Project Structure
```
Directory structure:
└── hariharans24-hybrid-ann-pca-mlp-model-for-accurate-credit-card-fraud-detection/
    ├── README.md
    ├── app.py
    ├── evaluate.py
    ├── gitignore.txt
    ├── LICENSE.txt
    ├── main.py
    ├── model.py
    ├── preprocess.py
    ├── requirements.txt
    ├── utils.py
    ├── data/
    │   └── dataset.txt
    └── models/
        ├── fraud_model.h5
        ├── pca_transformer.pkl
        └── scaler.pkl
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HARIHARANS24/Hybrid-ANN-PCA-MLP-Model-for-Accurate-Credit-Card-Fraud-Detection.git
cd Hybrid-ANN-PCA-MLP-Model-for-Accurate-Credit-Card-Fraud-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 💻 Usage
1. Launch the application using the command above
2. Upload or input transaction data
3. The model will process the data and provide fraud detection results
4. View detailed analysis and visualizations in the dashboard

## 📊 Model Architecture
- PCA Layer: Reduces feature dimensionality while preserving important information
- ANN Layers: Multiple dense layers with ReLU activation
- Output Layer: Sigmoid activation for binary classification
- Training: Adam optimizer with binary cross-entropy loss

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 👥 Authors
- **HARIHARANS24** - *Initial work* - [GitHub Profile](https://github.com/HARIHARANS24)

## 🙏 Acknowledgments
- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for their valuable tools and libraries

## 📞 Contact
For any queries or suggestions, please feel free to reach out through GitHub issues or create a pull request.

---
⭐ Star this repository if you find it helpful!













