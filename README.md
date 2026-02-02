# Iris Dataset — Supervised Classification and Unsupervised Structure Analysis

This project explores the classic **Iris dataset** through two complementary machine learning perspectives:

- **Supervised learning**, using logistic regression to classify flower species  
- **Unsupervised learning**, using Principal Component Analysis (PCA) and clustering to explore underlying structure  

By applying both approaches to the same dataset, the project highlights an important distinction in machine learning:  
**classification focuses on decision boundaries, while unsupervised learning focuses on variance and structure**.

Rather than treating the Iris dataset as a toy example, the analysis emphasises **careful problem framing, disciplined evaluation, and critical interpretation of results**.

![Photographs of the three iris species included in the dataset.](iris-machinelearning.png)

<sub>Image source: https://www.datacamp.com/tutorial/machine-learning-in-r</sub>

---

## What’s in this repository

- **Jupyter Notebooks:**  
  - Logistic regression classification (`iris_logistic_regression.ipynb`)  
  - PCA and clustering analysis (`PCA_task.ipynb`)  
- **Source data:** Iris dataset (`Iris.csv`)  
- **Images:** visualisations  
- **Requirements:** Python dependencies (`requirements.txt`)  

---

## Project Context

The Iris dataset contains four numerical measurements describing flower morphology:

- sepal length  
- sepal width  
- petal length  
- petal width  

alongside labels for three species:

- *Iris setosa*  
- *Iris versicolor*  
- *Iris virginica*  

The dataset is well suited to both supervised and unsupervised learning:

- clear class labels enable classification  
- strong correlations between features motivate dimensionality reduction  

This project deliberately examines **what each modelling paradigm can — and cannot — reveal**.

---

## Approach Overview

The analysis followed two distinct but related paths:

- Supervised classification using logistic regression  
- Unsupervised exploration using PCA and clustering  

Together, these approaches provide a rounded understanding of the dataset’s structure and separability.

---

## Supervised Learning: Logistic Regression Classification

### Problem Framing

The classification task was initially framed as a **binary problem**:

- **0:** *Iris setosa*  
- **1:** Not *Iris setosa*  

This allowed focused exploration of logistic regression as a probabilistic classifier. The analysis was later extended to **three-class classification** to evaluate how well the model distinguishes between all species.

---

### Modelling and Evaluation

The supervised workflow included:

- Feature selection using the four morphological measurements  
- Target encoding for binary and multiclass classification  
- Train–test splitting to enable unbiased evaluation  
- Model training using scikit-learn’s `LogisticRegression`  

Model performance was evaluated using:

- **Confusion matrices**  
- **Manual calculation** of:
  - accuracy  
  - precision  
  - recall  

These manually derived metrics were cross-checked against scikit-learn’s built-in functions to verify correctness and strengthen conceptual understanding.

![3x3 confusion matrix showing misclassification between versicolor and virginica.](iris-confusion-matrix.png)

The confusion matrix revealed that while *setosa* is easily separable, **versicolor and virginica overlap**, leading to occasional misclassification — an insight consistent with known properties of the dataset.

---

## Unsupervised Learning: PCA and Clustering

### Motivation for PCA

Exploratory analysis confirmed strong correlations between original features, particularly between petal length and petal width. This made the dataset well suited to **Principal Component Analysis**, which aims to:

- remove redundancy  
- transform correlated variables into independent components  
- preserve variance in fewer dimensions  

---

### Principal Component Analysis

After scaling the data, PCA was applied with **three principal components**.

![Biplot showing eigenvectors for original features.](iris-biplot.png)

Key observations included:

- Petal length and petal width eigenvectors were closely aligned, confirming strong correlation  
- Sepal measurements contributed more independently  
- The resulting principal components were **uncorrelated**, demonstrating successful dimensionality reduction  

Analysis of the transformed data showed that:

- PC1 retained a **bimodal distribution**, reflecting structure inherited from petal measurements  
- Subsequent components captured decreasing variance  
- PCA preserved variance, but not necessarily class separation  

---

### Clustering Analysis

Both **k-means clustering** and **hierarchical clustering** were applied to the PCA-transformed data.

Key findings:

- Clusters **did not align with the three species labels**  
- Dimensionality reduction did not improve separability for unsupervised clustering  
- This outcome illustrates a fundamental principle:
  - PCA preserves variance, not class boundaries  
  - Unsupervised learning identifies structure without reference to labels  

Rather than indicating failure, this result demonstrates **appropriate interpretation of unsupervised learning outcomes**.

---

## Key Insights / Findings

- Logistic regression performs well when class boundaries are clear, but struggles where species overlap  
- Manual metric calculation reinforces understanding of model evaluation  
- PCA successfully removes correlation and produces independent components  
- Dominant variance is driven by petal measurements  
- Unsupervised clustering does not recover species labels, highlighting the limits of variance-based grouping  
- Supervised and unsupervised methods answer fundamentally different questions  

---

## Skills Demonstrated

**Analysis**
- Exploratory data analysis  
- Correlation analysis and interpretation  

**Modelling**
- Logistic regression classification  
- Principal Component Analysis (PCA)  
- K-means clustering  
- Hierarchical clustering  

**Evaluation**
- Confusion matrix interpretation  
- Manual calculation of accuracy, precision, and recall  
- Critical assessment of unsupervised learning outcomes  

**Tools**
- Python  
- pandas  
- scikit-learn  
- matplotlib / seaborn  
- Jupyter Notebook  

---

## Requirements

Install the required Python packages with: `pip install -r requirements.txt`

---

## Why this project belongs in my portfolio
This project demonstrates the ability to apply multiple machine learning paradigms to the same dataset and interpret their results appropriately.

By contrasting supervised classification with unsupervised structure discovery, it shows:
- strong conceptual understanding of model intent
- disciplined evaluation rather than result chasing
- awareness of the limitations of common techniques

Together, these analyses reflect the kind of judgement required when applying machine learning methods in real-world settings.

Install the required Python packages with: `pip install -r requirements.txt`

