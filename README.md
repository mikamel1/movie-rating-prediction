# Movie Rating Prediction – Final Assignment Part 2

## Project Description

This project builds a Machine Learning model for predicting the average IMDb movie rating (`averageRating`) before the movie is released.

The model uses pre-release movie metadata such as release year, runtime, genres, languages, countries, plot availability, budget availability, and lead actor information.
Fields that are only available after release, such as `numVotes` and `BoxOffice`, were not used in order to prevent `data leakage`.

## Files in the Repository

* `final_assignment_part2.ipynb.ipynb` – the main Jupyter Notebook containing the full analysis, feature engineering, model training, evaluation, error analysis, and fairness analysis.
* `model.pkl` – the final trained model saved using `joblib`.
* `requirements.txt` – list of required Python libraries and versions.
* `README.md` – instructions for running the project.

## Requirements

The required Python libraries are listed in `requirements.txt`.

To install them, run:

```bash
pip install -r requirements.txt
```

The main libraries used in this project are:

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `scipy`
* `joblib`

## How to Run the Project

1. Clone or download the repository.

2. Make sure the dataset file is located in the same folder as the notebook.

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Open the notebook:

```bash
jupyter notebook final_assignment_part2.ipynb.ipynb
```

5. Run the notebook cells from top to bottom.

The notebook performs the following steps:

* Load and explore the raw movie dataset.
* Clean and inspect the data.
* Create engineered features using `prepare_data(df)`.
* Build preprocessing pipelines using imputation, scaling, and encoding.
* Train and evaluate two models using `10-fold cross-validation`.
* Compare model performance using `RMSE`, `MAE`, and `R²`.
* Analyze feature importance.
* Perform `Error Analysis`.
* Perform `Fairness Analysis`.
* Save the final trained model as `model.pkl`.

## Final Model

The final selected model is `Elastic Net`.

`Elastic Net` was selected because it achieved slightly better cross-validation performance compared to the second model, `DecisionTreeRegressor`.

The best `Elastic Net` parameters were:

* `alpha = 0.001`
* `l1_ratio = 0.1`

The final model is stored in:

```text
model.pkl
```

## Model Evaluation

The models were evaluated using `10-fold cross-validation`.

The main evaluation metrics were:

* `RMSE`
* `MAE`
* `R²`

The final `Elastic Net` model achieved approximately:

* `RMSE ≈ 1.118`
* `MAE ≈ 0.854`
* `R² ≈ 0.249`

## Data Leakage Prevention

The following columns were excluded from the model input:

* `averageRating` – target variable.
* `numVotes` – available only after users rate the movie.
* `BoxOffice` – available only after the movie is released.

All preprocessing steps that learn from the data, such as imputation, scaling, and encoding, were performed inside a `Pipeline` and therefore inside the cross-validation process.

## Using the Saved Model

The saved model expects data after applying the `prepare_data(df)` function.

Example usage:

```python
import joblib
import pandas as pd

model = joblib.load("model.pkl")

df_new = pd.read_csv("dataset chen.csv")
X_new = prepare_data(df_new)

predictions = model.predict(X_new)
```

Note: the function `prepare_data(df)` is defined in the notebook and must be available before using the saved model on new raw data.

## Notes

The model is intended for academic use as part of the Machine Learning final assignment.
Predictions should be interpreted carefully, especially for movies with missing metadata, non-US or unknown country information, and genres where the model showed higher error, such as `Action`.

