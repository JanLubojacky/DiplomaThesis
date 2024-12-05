import torch


def load_model(model, load_path: str):
    """Load a saved model state.

    Args:
        model: The initialized model architecture that matches the saved state
        load_path: Path to the saved model file
    """
    # Load the saved dictionary
    checkpoint = torch.load(load_path)

    # Load the state dictionary into your model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Optionally retrieve the best validation score
    best_val_score = checkpoint["best_val_score"]

    return model, best_val_score
