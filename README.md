# 🛒 Customer Segmentation — K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Unsupervised Machine Learning** project that segments mall customers into distinct groups based on their spending patterns and income levels using **K-Means Clustering**.

---

## ✨ Features

- 📊 **Interactive EDA** — Distributions, correlations, pair plots with Plotly
- 🔬 **Model Training** — Elbow Method & Silhouette Analysis for optimal K
- 🔮 **2D & 3D Visualizations** — Interactive scatter plots with cluster centroids
- 🎯 **Real-time Prediction** — Enter customer details to predict their segment
- 📋 **Cluster Profiles** — Business-friendly labels with marketing recommendations
- 🎨 **Premium Dark UI** — Glassmorphism design with smooth animations
- 🕸️ **Radar Charts** — Compare cluster characteristics at a glance

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ayanfaruqui/Customer-Segmentation.git
cd Customer-Segmentation

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Customer-Segmentation/
├── data/
│   └── Mall_Customers.csv              # Dataset (200 customers)
├── notebooks/
│   └── customer_segmentation.ipynb     # EDA + Model training notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning & feature engineering
│   ├── model.py                        # K-Means training & evaluation
│   └── visualizations.py              # Interactive Plotly visualizations
├── models/
│   ├── kmeans_model.pkl                # Trained K-Means model
│   └── scaler.pkl                      # Fitted StandardScaler
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
└── LICENSE                             # MIT License
```

---

## 📊 Dataset

The **Mall Customer Segmentation Dataset** contains 200 records with 5 features:

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male or Female |
| `Age` | Customer age (18-70) |
| `Annual Income (k$)` | Yearly income in thousands |
| `Spending Score (1-100)` | Mall-assigned behavior score |

**Source:** [Kaggle — Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## 🤖 Algorithm: K-Means Clustering

K-Means is an **unsupervised learning** algorithm that partitions data into K distinct clusters:

1. **Initialize** K centroids using k-means++ for better convergence
2. **Assign** each data point to the nearest centroid
3. **Update** centroids as the mean of assigned points
4. **Repeat** until convergence (centroids stabilize)

### Finding Optimal K

| Method | Description |
|---|---|
| **Elbow Method** | Plot WCSS vs K — look for the "elbow" bend |
| **Silhouette Score** | Measures cluster separation — higher is better |

For this dataset, **K = 5** produces the most meaningful customer segments.

---

## 🎯 Customer Segments Discovered

| Segment | Income | Spending | Strategy |
|---|---|---|---|
| 💰 **High Income, High Spenders** | High | High | VIP loyalty, exclusive launches |
| 📉 **High Income, Low Spenders** | High | Low | Personalized offers, quality focus |
| 🛍️ **Low Income, High Spenders** | Low | High | Rewards, flash sales, discounts |
| 💵 **Low Income, Low Spenders** | Low | Low | Budget essentials, basic loyalty |
| 🎯 **Moderate Spenders** | Medium | Medium | Seasonal deals, bundle offers |

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

 Deploy — https://customer-segmentation-machine-learning-032.streamlit.app/🎉



## 🛠️ Tech Stack

- **Python 3.10+** — Core language
- **Streamlit** — Web framework
- **scikit-learn** — K-Means Clustering
- **Plotly** — Interactive visualizations
- **Pandas & NumPy** — Data manipulation
- **Joblib** — Model serialization

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ayan Faruqui**  
- GitHub: [@ayanfaruqui](https://github.com/ayanfaruqui)

---

<p align="center">
  Made with ❤️ using Python, Streamlit & scikit-learn
</p>
