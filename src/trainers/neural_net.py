from __future__ import annotations

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from .base import ModelTrainer
from typing import Any
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


class TabularNNModule(nn.Module):

    '''
    PyTorch Tabular Deep Neural Network with Linear layers, BatchNorm1d, SiLU activation, and Dropout.
    '''

    def __init__(self, in_features: int, hidden_dims: list[int] | None = None, dropout: float = 0.25):
        
        super().__init__()
        dims = hidden_dims or [128, 64, 32]
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h_dim in dims:
            layers.extend([
                                nn.Linear(prev_dim, h_dim),
                                nn.BatchNorm1d(h_dim),
                                nn.SiLU(),
                                nn.Dropout(dropout)
                            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralNetTrainer(ModelTrainer):
   
    '''
    Specialized trainer for Tabular Neural Networks using PyTorch.
    '''

    model_preset_key: str = "neural_net"
    default_name: str = "TabularNeuralNet"


    def __init__(self, *args, device: str | None = None, **kwargs):

        if device is not None:
            self.device = torch.device(device)
        
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.nn_module: TabularNNModule | None = None
        super().__init__(*args, **kwargs)


    def _init_model(self, params: dict[str, Any]) -> Any:

        in_features = self.X_train.shape[1] if not self.X_train.empty else 10
        hidden_dims = list(params.get("hidden_layer_sizes", [128, 64, 32]))
        dropout = float(params.get("dropout", 0.25))
        self.nn_module = TabularNNModule(in_features = in_features, hidden_dims = hidden_dims, dropout = dropout).to(self.device)

        return self.nn_module


    def fit(self, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4,
            patience: int = 5, verbose: bool = False, **kwargs: Any) -> NeuralNetTrainer:
        
        '''
        It trains the PyTorch neural network, logging exact per-epoch training and validation loss curves.

        Parameters
        ----------
        epochs: int
            Maximum number of training epochs [Default = 30].

        batch_size: int
            Batch size for DataLoader mini-batches [Default = 64].

        lr: float
            Initial learning rate for the AdamW optimizer [Default = 1e-3].

        weight_decay: float
            L2 regularization penalty [Default = 1e-4].

        patience: int
            Number of consecutive epochs without validation accuracy improvement before early stopping [Default = 5].

        verbose: bool
            Whether to print training progress and early stopping messages [Default = False].

        **kwargs: Any
            Additional keyword arguments (unused, for interface consistency).

        Returns
        -------
        NeuralNetTrainer
            The fitted trainer instance.
        '''

        torch.manual_seed(self.random_state)
        self._init_model(self.params)

        if self.nn_module is None:
            raise RuntimeError("Neural network module failed to initialize.")

        train_loader = self._prepare_train_loader(batch_size = batch_size)
        val_data = self._prepare_val_data()

        opt, sched, crit = self._init_training_components(lr = lr, weight_decay = weight_decay)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0
        best_weights = None
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):

            tr_loss, tr_acc = self._train_epoch(train_loader = train_loader, opt = opt, crit = crit)
            train_losses.append(tr_loss)
            train_accs.append(tr_acc)

            if val_data is not None:

                v_loss, v_acc = self._validate_epoch(val_data = val_data, crit = crit)
                val_losses.append(v_loss)
                val_accs.append(v_acc)
                sched.step(v_acc)

                if v_acc > best_val_acc:
                    best_val_acc = v_acc
                    best_weights = {k: v.cpu().clone() for k, v in self.nn_module.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_weights is not None:
            self.nn_module.load_state_dict(best_weights)

        self._record_history(train_losses = train_losses, val_losses = val_losses, train_accs = train_accs, val_accs = val_accs)
        self.model = self.nn_module
        self.is_fitted = True

        return self


    def _prepare_train_loader(self, batch_size: int) -> DataLoader:

        '''
        It creates a PyTorch DataLoader for the training dataset.

        Parameters
        ----------
        batch_size: int
            Batch size for DataLoader mini-batches.

        Returns
        -------
        DataLoader
            Configured training DataLoader.
        '''

        X_tr_t = torch.tensor(self.X_train.to_numpy(), dtype = torch.float32)
        y_tr_t = torch.tensor(self.y_train.to_numpy(), dtype = torch.float32)

        return DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size = batch_size, shuffle = True)


    def _prepare_val_data(self) -> tuple[torch.Tensor, np.ndarray] | None:

        '''
        It converts validation data into a PyTorch tensor on the target device and a NumPy target array.

        Returns
        -------
        tuple[torch.Tensor, np.ndarray] | None
            Tuple of (X_val_tensor, y_val_numpy) if validation set is non-empty, otherwise None.
        '''
        
        if self.X_val.empty:
            return None

        X_val_t = torch.tensor(self.X_val.to_numpy(), dtype = torch.float32).to(self.device)
        y_val_np = self.y_val.to_numpy()

        return X_val_t, y_val_np


    def _init_training_components(self, lr: float, 
                                  weight_decay: float) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau, nn.BCEWithLogitsLoss]:

        '''
        It initializes the optimizer, learning rate scheduler, and loss criterion.

        Parameters
        ----------
        lr: float
            Initial learning rate.

        weight_decay: float
            L2 regularization penalty.

        Returns
        -------
        tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau, nn.BCEWithLogitsLoss]
            Optimizer, scheduler, and loss criterion.
        '''

        if self.nn_module is None:
            raise RuntimeError("Neural network module is not initialized.")

        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(self.nn_module.parameters(), lr = lr, weight_decay = weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "max", factor = 0.5, patience = 2)

        return opt, sched, crit


    def _train_epoch(self, train_loader: DataLoader, opt: torch.optim.Optimizer, crit: nn.Module) -> tuple[float, float]:

        '''
        It runs a single training epoch across all mini-batches.

        Parameters
        ----------
        train_loader: DataLoader
            DataLoader containing training batches.

        opt: torch.optim.Optimizer
            Optimizer instance.

        crit: nn.Module
            Loss criterion.

        Returns
        -------
        tuple[float, float]
            Tuple of (average_train_loss, train_accuracy).
        '''

        if self.nn_module is None:
            raise RuntimeError("Neural network module is not initialized.")

        self.nn_module.train()
        running_loss = 0.0
        all_preds, all_targets = [], []

        for bx, by in train_loader:

            bx, by = bx.to(self.device), by.to(self.device)
            opt.zero_grad()
            logits = self.nn_module(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(by)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs > 0.5)
            all_targets.extend(by.cpu().numpy())

        epoch_loss = float(running_loss / len(self.X_train))
        epoch_acc = float(accuracy_score(all_targets, all_preds))

        return epoch_loss, epoch_acc


    def _validate_epoch(self, val_data: tuple[torch.Tensor, np.ndarray], crit: nn.Module) -> tuple[float, float]:

        '''
        It evaluates the neural network on the validation set.

        Parameters
        ----------
        val_data: tuple[torch.Tensor, np.ndarray]
            Tuple of (X_val_tensor, y_val_numpy).

        crit: nn.Module
            Loss criterion.

        Returns
        -------
        tuple[float, float]
            Tuple of (validation_loss, validation_accuracy).
        '''
        if self.nn_module is None:
            raise RuntimeError("Neural network module is not initialized.")

        X_val_t, y_val_np = val_data
        self.nn_module.eval()

        with torch.no_grad():

            v_logits = self.nn_module(X_val_t)
            v_loss = crit(v_logits, torch.tensor(y_val_np, dtype = torch.float32).to(self.device)).item()
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
            v_acc = accuracy_score(y_val_np, v_probs > 0.5)

        return float(v_loss), float(v_acc)


    def _record_history(self, train_losses: list[float], val_losses: list[float], train_accs: list[float], val_accs: list[float]) -> None:

        '''
        It records training and validation loss and accuracy curves into self.history.

        Parameters
        ----------
        train_losses: list[float]
            List of training losses per epoch.

        val_losses: list[float]
            List of validation losses per epoch.

        train_accs: list[float]
            List of training accuracies per epoch.

        val_accs: list[float]
            List of validation accuracies per epoch.
        '''

        self.history["train_loss"] = train_losses
        self.history["val_loss"] = val_losses
        self.history["train_metric"] = train_accs
        self.history["val_metric"] = val_accs
        self.history["iterations"] = list(range(1, len(train_losses) + 1))


    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:

        '''
        It generates positive class probability predictions for input samples.

        Parameters
        ----------
        X: pd.DataFrame | np.ndarray
            Input features to generate predictions for.

        Returns
        -------
        np.ndarray
            Predicted probabilities for class 1.
        '''

        if not self.is_fitted or self.nn_module is None:
            raise RuntimeError("NeuralNetTrainer must be fitted before predict_proba().")

        self.nn_module.eval()
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        tensor_x = torch.tensor(arr, dtype = torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.nn_module(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs