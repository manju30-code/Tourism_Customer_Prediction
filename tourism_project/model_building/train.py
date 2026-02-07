    # Save the model locally
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Save the preprocessor
    preprocessor_path = "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "mkrish2025/Tourism-Customer-Prediction"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_prediction_model_v1.joblib",
        path_in_repo="best_tourism_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

    api.upload_file(
        path_or_fileobj="preprocessor.joblib",
        path_in_repo="preprocessor.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
