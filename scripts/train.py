from house_price_prediction.config import load_settings


def _resolve_trainer(model_type: str):
    if model_type == "lightgbm":
        from house_price_prediction.model import train_and_save_model as lightgbm_trainer

        return lightgbm_trainer
    if model_type in {"random_forest", "random-forest", "rf"}:
        from house_price_prediction.model_random_forest import (
            train_and_save_model as random_forest_trainer,
        )

        return random_forest_trainer
    raise ValueError(
        f"Unsupported MODEL_TYPE '{model_type}'. Use 'lightgbm' or 'random_forest'."
    )


if __name__ == "__main__":
    settings = load_settings()
    train_and_save_model = _resolve_trainer(settings.model_type)
    metrics = train_and_save_model(settings)

    print("Training complete")
    print(f"Saved model: {settings.model_path}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")