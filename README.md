# Genre Classification

The Machine Learning pipeline with MLflow 🤖

## Run

Given you have navigated into the root folder of this project. This will run the complete pipeline.

```bash
mlflow run .
```


Override pipeline steps and only run `random_forest` step.
```bash
mlflow run . -P hydra_options="main.execute_steps='random_forest'"
```


### Run from GitHub

If you are logged into [Weight & Biases](https://wandb.ai) you can run this MLflow pipeline from this GitHub repository.

```bash
mlflow run -v 1.0.0 https://github.com/marcusholmgren/genre_classification
```
