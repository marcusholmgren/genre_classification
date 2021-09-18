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