# Machine Learning Project

Final Project for SJSU CS-171 Section 2

Authors: [Jonathan Nguyen](https://github.com/jonathanguven) and [Brandon Nguyen](https://github.com/brandonnguyenn27)

## Description

This project investigates the relationship between housing features and sale price to understand which factors influence home values the most. It uses datasets of residential properties that include attributes like lot size, the number of rooms, the year it was built, and its location. We aim to develop machine learning models that can predict house prices with high accuracy. We will explore both simple and complex regression approaches to model these relationships and analyze their predictive power.

Research question: *How accurately can we predict a home's price based on its physical characteristics and location using supervised learning methods?*

## Project Outline

### Data Collection Plan

Jonathan Nguyen
- Obtain the house prices using the advanced regression techniques dataset from Kaggle or other open-source datasets.
- Perform data cleaning (handling missing values, removing outliers, and verifying data).
- Implement feature encoding for variables such as neighborhoot and use one-hot encoding.
- Normalize or standardize features like square footage, lot area, and year built to prepare them for the ML models.
- Split the data into training, testing, and validation sets for reproducibility.

Brandon Nguyen
- Find and prepare an open-source data set of house prices in California.
- Clean the data to prepare it for model training, and normalize/encode attributes
- Perform exploratory data analysis to identify patterns and correlations between features and price.

### Model Plans
Jonathan Nguyen - Linear and Ensemble Models
- Implement Multiple Linear Regression as a baseline model to estimate price as a function of multiple predictors.
- Evaluate regularized models such as Ridge or Lasso Regression to handle multicollinearity and overfitting.
- Include an Ensemble Learning approach to compare performance against linear methods.
- Evaluate results using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score, and visualize prediction accuracy with plots of predicted vs. actual prices.

Brandon Nguyen - Neural Network Model (MLP with PyTorch)
- Build a Multilayer Perceptron (MLP) regression model using PyTorch, incorporating at least one hidden layer with nonlinear activation functions (ReLU).
- Train the network using the preprocessed data and compare its learning curve to that of traditional models.
- Tune hyperparameters such as learning rate, number of epochs, and layer size to minimize loss and prevent overfitting.
- Visualize loss convergence across epochs and compare the model’s predictive performance with the linear and ensemble baselines to analyze improvements in capturing non-linear relationships.

## Project Timeline

### Week 1: Project Definition and Finding Datasets

**Goals**

- Determine **research question** / problem statement.
- Select **two real-world datasets** (one per partner) to support the machine-learning modeling.
- Discuss / finalize dataset choices with each other.

**Deliverables**

- Create GitHub repository and project structure
- Complete GitHub README file.

### Week 2: Data Pre-processing Notebooks

**Goals**

- Each partner writes their **data preprocessing notebook**
- Includes: data cleaning, transformations, feature engineering, and train/test/validation splits
- Confirm reproducibility
- Document each step with Markdown cells

**Deliverables**

- Completed data preprocessing notebook for each partner's dataset
- Dataset ready for model training

### Week 3: Model Construction and Training

**Goals**

- Design and train **one model per partner**.
- Record hyperparameters and evaluation metrics
- Record and visualize model performance on training sets

**Deliverables**

- Completed model construction notebook with model training and evaluation
- Reproducibility test on both partner's machines

### Week 4: Analysis and Visualization Notebooks

**Goals**

- Analyze performance of trained model against validation dataset with plots and charts.
- Tie visualizations to insights related to the research question
- Refine model as necessary

**Deliverables**

- Completed analysis notebook with comments, code, and Markdown cells complete.
- Verify reproducibility for different environments.

### Week 5: Final Review and Presentation Preparation

**Goals**

- Polish all notebooks
- Continue to verify reproducibility for clean environments
- Finalize slides for **Dec2-4 presentations**
- Practice live demo of notebooks and visualizations

**Deliverables**

- Final GitHub repository with completed notebooks and datasets, ready for submission
- Polished presentation slides.
