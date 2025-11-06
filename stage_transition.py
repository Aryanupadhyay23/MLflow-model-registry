from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Define the model name and version
client = MlflowClient()

# Define the model name and version
model_name = "diabetes-rf"
model_version = 5 # Replace with the specific version number you want to transition

# Transition the model version to a new stage
new_stage = "Production" # Possible values: "None", "Staging", "Production", "Archived"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model version {model_version} transitioned to {new_stage}")

