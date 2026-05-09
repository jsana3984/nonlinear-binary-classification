# nonlinear-binary-classification
# Non-Linear Binary Classification & Probability Estimation

## Overview
This project tackles a binary classification problem where a purely linear decision boundary is insufficient. It explores the feature space, engineers non-linear polynomial and interaction terms, and rigorously compares Logistic Regression against Quadratic Discriminant Analysis (QDA) to optimize for accuracy, recall, and log-loss.

A core component of this project involves estimating true class probabilities and calculating mathematically rigorous 95% confidence intervals on the log-odds scale using the model's asymptotic covariance matrix.

## Key Skills Demonstrated
* **Statistical Modeling:** Logistic Regression, Quadratic Discriminant Analysis (QDA).
* **Feature Engineering:** Polynomial feature expansion (degree 2) and interaction terms to capture complex underlying data structures.
* **Advanced Inference:** Extracting standard errors from the asymptotic covariance matrix of the coefficients to compute exact 95% confidence intervals, transforming them via the inverse-logit function.
* **Model Evaluation:** AUC-ROC, Precision/Recall, F1 Score, and Log-Loss optimization.
* **Programmatic Reporting:** Automated generation of technical presentation slides with embedded visualizations using `matplotlib.backends.backend_pdf`.

## Repository Structure
```text
├── data/                   # Training and testing datasets
├── src/                    
│   ├── model_training.py   # Primary analysis, model fitting, and CI generation
│   └── generate_slides.py  # Code to programmatically generate the PDF report
├── docs/                   
│   └── Analysis_Presentation.pdf # Automated 3-slide visual summary
├── output/                 
│   └── predictions.csv     # Final probability estimates and confidence bounds
└── README.md
## Methodology & Results

**Exploratory Data Analysis:** Initial analysis revealed a non-linear relationship between the predictors and the response, prompting the introduction of degree-2 polynomial features to capture higher-order dependencies, such as $x_i^2$ and $x_i x_j$.

**Model Comparison:** * **Quadratic Discriminant Analysis (QDA)** achieved a strong overall ranking capability with an $AUC \approx 0.799$. However, it suffered from a high false-negative rate at the default classification threshold, which severely impacted recall ($51.7\%$).
* **Logistic Regression** (incorporating engineered interaction terms) proved vastly superior at the default threshold. The model achieved a balanced confusion matrix, an accuracy of $76.6\%$, and an $F_1$ score of $0.77$.

**Conclusion:** Logistic Regression outperformed QDA in this specific context. While QDA relies on strict normality assumptions for predictors to construct quadratic boundaries, the engineered Logistic Regression model flexibly captured the non-linear boundary without being constrained by such $i.i.d.$ distributional assumptions.

## Technologies Used
* **Python**: Primary programming language for implementation.
* **statsmodels**: Utilized for precise statistical modeling and extracting the asymptotic covariance matrix $\hat{\Sigma}$ for 95% confidence interval estimation.
* **scikit-learn**: Used for model evaluation metrics and QDA baseline implementation.
* **pandas & numpy**: Essential for data manipulation and numerical computation.
* **matplotlib & seaborn**: Employed for statistical data visualization and diagnostic plotting.
