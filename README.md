# ⚽ Football Match Outcome Prediction with EDA & Machine Learning

## 📌 Project Overview
This project explores **football match data** using **Exploratory Data Analysis (EDA)** and applies **Machine Learning (ML)** models to predict match outcomes. We analyze various **match statistics, team performances, and seasonal trends** to gain insights into the factors affecting match results.

## 📂 Dataset
We use the **European Soccer Database** available on **Kaggle**:
[📂 Soccer Dataset](https://www.kaggle.com/datasets/hugomathien/soccer)

The dataset contains information on:
- **Matches** (scores, teams, dates, statistics)
- **Teams & Attributes** (offensive & defensive metrics)
- **Players & Attributes** (skills, positions, performance stats)
- **Leagues & Countries**

## 🔍 Exploratory Data Analysis (EDA)
### Key Questions Explored:
1. **What factors influence match outcomes?**
2. **Are there seasonal patterns in wins, goals, or tournament types?**
3. **Which teams and players consistently outperform others?**
4. **Are there anomalies, such as extreme score differences?**
5. **Do home teams have a statistical advantage over away teams?**

## 📊 Key Insights
1. **Match Statistics:**
   - Home teams generally perform better, winning around **54%** of matches.
   - **Possession, shots on target, and corners** strongly correlate with winning outcomes.
2. **Seasonal Trends:**
   - Goal-scoring rates tend to be higher in **warmer months (April–June)**.
   - League tournaments show more consistent results compared to knockout tournaments.
3. **Top Performing Teams & Players:**
   - Certain teams consistently rank high in win rates and goal-scoring ability.
   - Player attributes like **dribbling, finishing, and acceleration** play key roles in attacking success.
4. **Anomaly Detection:**
   - Some matches exhibit **extreme goal differences** (e.g., 7-0 results).
   - Red cards and early-game injuries significantly impact match outcomes.
5. **Home vs. Away Performance:**
   - Home teams have a **higher possession rate** and take more shots on target.
   - Away teams rely more on counter-attacks and defensive strategies.

## ⚙️ Machine Learning Pipeline
1. **Preprocessing**: Handling missing values, encoding categorical data, and normalizing numerical features.
2. **Feature Engineering**: Selecting relevant match & team attributes.
3. **Model Training**: Training multiple classification models:
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
   - Support Vector Machine (SVM)
4. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

## 🛠️ Installation & Setup
### Prerequisites
Ensure you have **Python 3.7+** and install required dependencies:
```sh
pip install -r requirements.txt
```

### Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/soccer-eda-ml.git
   cd soccer-eda-ml
   ```
2. Run the data analysis script:
   ```sh
   python eda_analysis.py
   ```
3. Train the machine learning models:
   ```sh
   python train_models.py
   ```
4. Evaluate the models:
   ```sh
   python evaluate_models.py
   ```

## 📊 Visualizations
- **Feature correlations with match results**
- **Trends in match statistics across seasons**
- **Home vs. away team performance comparisons**
- **Top teams based on win rates and goals scored**

## 🚀 Results & Insights
- **Possession, shots on target, and corners are key predictors of match outcomes.**
- **Home teams tend to win more frequently than away teams.**
- **Defensive and offensive metrics significantly impact team performance.**
- **Random Forest and XGBoost outperform Logistic Regression in accuracy.**
