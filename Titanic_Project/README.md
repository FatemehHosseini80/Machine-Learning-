
# 🚢 Titanic Survival Analysis & EDA

An Exploratory Data Analysis (EDA) project focused on the classic Titanic dataset. This project investigates the demographics and passenger attributes to uncover patterns and factors that influenced survival rates during the tragedy.

## 🎯 Project Highlights
* **Data Cleaning & Preprocessing:** Handled missing data effectively by imputing `Age` with the mean and `Embarked` with the mode. Removed non-predictive columns (`Name`, `Ticket`, `Cabin`) to streamline the dataset.
* **Feature Engineering:** Created new meaningful variables such as `FamilySize` (combining `SibSp` and `Parch`) and a boolean `IsAlone` feature to analyze the impact of family presence on survival.
* **Statistical Analysis:** Extracted key insights, revealing an overall survival rate of 38.38%, with a stark contrast between female (74.2%) and male (18.89%) passengers, as well as significant differences across ticket classes.
* **Rich Data Visualization:** Leveraged graphical analysis to present findings, including:
  * Histograms with KDE for age distributions by survival status.
  * Boxplots comparing age, travel class, and survival.
  * Correlation heatmaps to identify relationships between numerical features.

## 📊 Dataset Overview
The dataset contains information on 891 Titanic passengers. Key features analyzed include:
* `Pclass`: Ticket class (1st, 2nd, 3rd)
* `Sex`: Passenger gender
* `Age`: Passenger age in years
* `SibSp` & `Parch`: Number of siblings/spouses and parents/children aboard
* `Fare`: Passenger fare
* `Embarked`: Port of Embarkation (C, Q, S)

## 🚀 Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 📈 Key Findings
1. **Gender:** Females had a significantly higher chance of survival compared to males.
2. **Socio-Economic Status:** First-class passengers had the highest survival rate (62.9%), which drastically dropped for third-class passengers (24.2%).
3. **Companionship:** Passengers traveling alone had a lower survival rate (~30.3%) compared to those traveling with family members.
