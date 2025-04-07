# 🧠 Stock Market Sentiment Analysis using News Headlines

This capstone project applies **Natural Language Processing (NLP)** and **deep learning (LSTM)** to predict stock market sentiment based on news headlines. The model is trained to classify whether a given headline signals a positive or negative stock movement.

---

## 📂 Project Structure
📁 Capstone_NLP/
│
├── Capstone_NLP.ipynb      # Main notebook with code for preprocessing, training & prediction
├── trained randomclassifier_model.h5   # Saved  model
├── data/
│   └── news.csv            # Input dataset with headlines and sentiment labels
├── README.md               # Project overview and instructions


---

## 🛠️ Features

- Cleans and preprocesses financial news headlines
- Vectorizes text using `CountVectorizer`
- Trains an **LSTM neural network** for sentiment classification
- Evaluates performance using precision, recall, F1-score
- Predicts sentiment of custom user-input headlines

---

## 🧪 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib / Seaborn  
- Google Colab / Jupyter  

---

## 🧩 Workflow

1. **Data Preprocessing**
   - Clean text (remove punctuation, lowercase, etc.)

2. **Vectorization**
   - Convert headlines into numeric form using Bag-of-Words

3. **Model Training**
   - RandomForest Classifier model trained on processed input

4. **Prediction**
   - Input custom headlines for sentiment prediction

---

## 📊 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.90  |
| Precision  | 0.89  |
| Recall     | 0.90  |
| F1-Score   | 0.89  |

> ✅ The model achieved **90% accuracy** on the test dataset.

---

## 🔁 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Capstone_NLP.git



