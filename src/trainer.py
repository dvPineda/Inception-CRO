# src/trainer.py

import os
import torch
import torch.nn as nn
from src.utils import evaluate_model

class Trainer:
    """
    Trainer class to encapsulate partial and full training logic for the MNIST Inception models.

    Attributes:
        device (torch.device): CPU or GPU.
        config (dict): Global configuration dictionary containing hyperparameters.
        criterion (nn.Module): Loss function (CrossEntropy).
    """
    def __init__(self, device, config):
        """
        Constructor for Trainer.

        Args:
            device (torch.device): 'cpu' or 'cuda'.
            config (dict): Configuration dictionary with training hyperparameters.
        """
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def partial_train(self, model, train_loader):
        """
        Train the model for a limited number of batches to obtain a quick estimation
        of its performance. This method is called during the CRO optimization to
        compute the fitness function efficiently.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): The loader providing training data.

        Returns:
            None
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        max_batches = self.config["num_batches"]
        batch_count = 0

        for images, labels in train_loader:
            if batch_count >= max_batches:
                break

            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count += 1

    def full_train(self, model, train_loader, val_loader):
        """
        Perform a full training routine (with early stopping) for the best model identified
        by CRO. Used after the evolutionary algorithm has finished.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): The loader providing training data.
            val_loader (DataLoader): The loader providing validation data.

        Returns:
            None
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        best_val_loss = float('inf')
        epochs_no_improve = 0

        patience = self.config["patience"]
        num_epochs = self.config["num_epochs"]

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            # Training loop
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # Evaluate on the validation set (to track val_loss)
            model.eval()
            fitness, accuracy, val_loss = evaluate_model(
                model,
                val_loader,
                self.device,
                fitness_method=self.config["fitness_method"],
                alpha=self.config["fitness_alpha"],
                beta=self.config["fitness_beta"]
            )

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0


                # Save checkpoint
                checkpoint_dir = self.config["checkpoints_dir"]
                os.makedirs(checkpoint_dir, exist_ok=True) # Ensure directory exists
                best_model_path = f"{checkpoint_dir}/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
