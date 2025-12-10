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
Brandon Nguyen - Linear and Ensemble Models
- Implement Multiple Linear Regression as a baseline model to estimate price as a function of multiple predictors.
- Evaluate regularized models such as Ridge or Lasso Regression to handle multicollinearity and overfitting.
- Include an Ensemble Learning approach to compare performance against linear methods.
- Evaluate results using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score, and visualize prediction accuracy with plots of predicted vs. actual prices.

Jonathan Nguyen - Neural Network Model (MLP with PyTorch)
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

# Running the Models

## Multi-Layer Perceptron - Jonathan Nguyen

### Data Access

The project uses the California Housing dataset. Ensure you have the following data files:
- `Multi-Layer Perceptron/data/california_housing_raw.csv` - Raw housing data for MLP model
- `ensemble/california_housing.csv` - Housing data for ensemble models

The MLP model workflow consists of three sequential notebooks that must be run in order.

### Notebook 1: Data Preprocessing (`NB1_Preprocessing.ipynb`)

**Location:** `Multi-Layer Perceptron/NB1_Preprocessing.ipynb`

**Purpose:** Cleans raw housing data and creates train/validation/test splits.

**Steps:**
1. Navigate to the `Multi-Layer Perceptron/` directory
2. Ensure `data/california_housing_raw.csv` exists
3. Run all cells in the notebook sequentially
4. The notebook will:
   - Load and inspect the raw data
   - Drop rows with missing values in key columns (price, bed, bath, acre_lot, street, city, state, zip_code, house_size)
   - Filter out extreme price outliers (99th percentile cap)
   - Split data into 70% train, 15% validation, and 15% test sets
   - Save cleaned splits to `processed/train.csv`, `processed/val.csv`, and `processed/test.csv`

**Output:**
- `data/california_housing_clean.csv` - Cleaned dataset
- `processed/train.csv` - Training split
- `processed/val.csv` - Validation split
- `processed/test.csv` - Test split

### Notebook 2: Model Training (`NB2_Model_Training.ipynb`)

**Location:** `Multi-Layer Perceptron/NB2_Model_Training.ipynb`

**Purpose:** Trains a Multilayer Perceptron regression model using PyTorch.

**Prerequisites:**
- Must run `NB1_Preprocessing.ipynb` first to generate data splits

**Steps:**
1. Navigate to the `Multi-Layer Perceptron/` directory
2. Run all cells in the notebook sequentially
3. The notebook will:
   - Load train/validation/test splits from `processed/` directory
   - Build a preprocessing pipeline (StandardScaler for numeric features, OneHotEncoder for categorical features)
   - Transform features and convert to PyTorch tensors
   - Define and train an MLP model with architecture: 2884 → 128 → 64 → 32 → 1
   - Train for 150 epochs with Adam optimizer (learning rate: 1e-3)
   - Save the trained model weights and preprocessing pipeline

**Output:**
- `artifacts/preprocessor.joblib` - Fitted preprocessing pipeline
- `artifacts/mlp_model_weights.pth` - Trained model weights
- Training and validation loss curves visualization

### Notebook 3: Model Evaluation (`NB3_Model_Evaluation.ipynb`)

**Location:** `Multi-Layer Perceptron/NB3_Model_Evaluation.ipynb`

**Purpose:** Evaluates the trained MLP model on the test set and generates visualizations.

**Prerequisites:**
- Must run `NB1_Preprocessing.ipynb` first
- Must run `NB2_Model_Training.ipynb` first to generate trained model

**Steps:**
1. Navigate to the `Multi-Layer Perceptron/` directory
2. Ensure `mlp.py` exists in the same directory (contains the MLPRegressor class definition)
3. Run all cells in the notebook sequentially
4. The notebook will:
   - Load the test data splits
   - Reload the saved preprocessor and model weights
   - Generate predictions on the test set
   - Compute regression metrics (MSE, RMSE, MAE, R²) in both scaled and real dollar units
   - Create visualizations of predicted vs actual prices and residual plots

**Output:**
- Regression metrics printed to console
- Visualization plots showing model performance

## Ensemble Models - Brandon Nguyen

### Ensemble Models Notebook (`cs171-project.ipynb`)

**Location:** `ensemble/cs171-project.ipynb`

**Purpose:** Implements and compares linear regression, regularized models (Ridge/Lasso), and ensemble methods for housing price prediction.

**Steps:**
1. Navigate to the `ensemble/` directory
2. Ensure `california_housing.csv` exists in the same directory
3. Run all cells in the notebook sequentially
4. The notebook will:
   - Load and preprocess the California housing data
   - Drop unnecessary columns and handle missing values
   - Perform feature engineering and encoding
   - Train multiple models:
     - Multiple Linear Regression (baseline)
     - Ridge Regression
     - Lasso Regression
     - Ensemble methods (e.g., Random Forest, Gradient Boosting)
   - Evaluate models using MSE, RMSE, and R² scores
   - Generate visualizations comparing model performance

**Output:**
- Trained model files in `models/` directory
- Performance comparison visualizations
- Model evaluation metrics

**Note:** The ensemble notebook is self-contained and does not require outputs from the MLP notebooks.

# Future Updates

If we were to continue working on this project, we would pursue the following improvements and extensions:

## Model Improvements
- **Hyperparameter Optimization**: Implement systematic hyperparameter tuning using GridSearchCV or Bayesian optimization for both MLP and ensemble models to find optimal configurations
- **Advanced Neural Architectures**: Experiment with deeper networks, residual connections, and attention mechanisms to capture more complex feature interactions
- **Feature Engineering**: Create additional features such as:
  - Price per square foot ratios
  - Distance-based features (proximity to schools, amenities)
  - Temporal features from `prev_sold_date` (time since last sale, market trends)
  - Interaction terms between key features
- **Ensemble of Ensembles**: Combine predictions from MLP and ensemble models using stacking or weighted averaging to leverage strengths of both approaches

## Data Enhancements
- **External Data Integration**: Incorporate additional features such as:
  - Census data (median income, population density)
  - School district ratings
  - Crime statistics
  - Walkability scores
  - Public transportation accessibility
- **Data Augmentation**: Generate synthetic samples for underrepresented price ranges or locations to improve model generalization
- **Outlier Handling**: Implement more sophisticated outlier detection methods (e.g., Isolation Forest, DBSCAN) instead of simple percentile capping

## Technical Improvements
- **Cross-Validation**: Implement k-fold cross-validation for more robust model evaluation and hyperparameter selection
- **Early Stopping**: Add early stopping mechanisms to prevent overfitting during MLP training
- **Model Interpretability**:
  - Implement SHAP values to understand feature importance
  - Create partial dependence plots to visualize feature effects
  - Generate model explanations for individual predictions
- **Deployment Pipeline**: Create a production-ready API or web interface for real-time price predictions

## Evaluation Enhancements
- **Additional Metrics**: Calculate MAPE (Mean Absolute Percentage Error), quantile loss, and prediction intervals
- **Error Analysis**: Perform detailed error analysis to identify patterns in prediction errors (e.g., by price range, location, property type)
- **A/B Testing Framework**: Set up framework to compare model versions on new data
- **Time-based Validation**: If temporal data is available, implement time-series cross-validation to test model performance on future data

## Documentation and Code Quality
- **API Documentation**: Add docstrings to all functions and classes following Python conventions
- **Unit Tests**: Write comprehensive unit tests for data preprocessing, model training, and evaluation functions
- **Tutorial Notebooks**: Create beginner-friendly tutorial notebooks explaining key concepts
- **Performance Benchmarks**: Document training times and resource requirements for different model configurations
