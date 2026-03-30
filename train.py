import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):

    # Print the loaded configuration
    print("=== Loaded Hydra Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    print("==================================\n")

    # Build the Model using config parameters
    print("Initializing Neural Network...")

    # Choose activation function dynamically based on config
    activation_layer = nn.ReLU() if cfg.model.activation == "relu" else nn.Tanh()

    model = nn.Sequential(
        nn.Linear(cfg.model.input_size, cfg.model.hidden_size),
        activation_layer,
        nn.Dropout(cfg.model.dropout),
        nn.Linear(cfg.model.hidden_size, 10)
    )
    print(model)

    # Set up the Optimizer
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    print("Initialized Optimizer:")
    print(optimizer)

    # Dummy Training Loop
    print(f"\nStarting training for {cfg.training.epochs} epochs (Batch Size: {cfg.training.batch_size})...")
    for epoch in range(1, cfg.training.epochs + 1):
        # Simulating a forward/backward pass
        dummy_loss = 2.0 / (epoch + (cfg.training.learning_rate * 100))

        print(f"Epoch [{epoch}/{cfg.training.epochs}] - Loss: {dummy_loss:.4f}")

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()