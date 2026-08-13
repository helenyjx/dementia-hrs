from nnblocks import MLP, GNN, GCN, GAT

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NeuralClassifier(nn.Module):
    def __init__(
            self, hidden_layer_sizes,
            max_iter=50, weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
            dropout_p=0.0, use_residual=True, use_batch_norm=False,
            batch_size=512, shuffle=True, random_state=42, cuda=True, loss_type='weighted_bce') -> None:
        super(NeuralClassifier, self).__init__()

        # Model configuration
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_size = None
        self.output_size = 1
        self.dropout_p = dropout_p
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.model = None
        self.criterion = None  # Will be set in fit() with proper weights
        self.loss_type = loss_type  # 'bce', 'weighted_bce', 'focal'

        # Optimizer configuration
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.squares = squares
        self.optimizer_t = optimizer_t

        # Environment configuration
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.cuda_ = cuda

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        X = X.to_numpy(dtype=np.float32)
        y = y.to_numpy(dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        self.input_size = X.shape[1]
        self._build_model()
        
        # Compute class weights for imbalanced data
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        
        if self.cuda_:
            pos_weight_tensor = pos_weight_tensor.cuda()
        
        # Set loss function based on loss_type
        if self.loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("Using Focal Loss")
        elif self.loss_type == 'weighted_bce':
            self.criterion = None  # Will compute manually
            print("Using Weighted BCE Loss")
        else:
            self.criterion = nn.BCELoss(reduction='mean')
            print("Using standard BCE Loss")
        
        print(f"Class distribution: Negative={neg_count} ({neg_count/len(y)*100:.2f}%), "
              f"Positive={pos_count} ({pos_count/len(y)*100:.2f}%)")
        if self.loss_type in ['weighted_bce', 'focal']:
            print(f"Positive class weight: {pos_weight:.2f}")

        # TODO: remove it to avoid out of memory
        # Keep it when dataset is small to make learning run faster
        # if self.cuda_:
        #     X_tensor = X_tensor.cuda()
        #     y_tensor = y_tensor.cuda()

        match self.optimizer_t:
            case 'sgdm':
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                    momentum=self.momentum, nesterov=True)
            case 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                    betas=(self.momentum, self.squares))
            case _:
                raise NotImplementedError(
                    "Optimizer '{}' not implemented.".format(self.optimizer_t))

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.model.train()

        for epoch in range(self.max_iter):
            with tqdm(loader, desc=f"Epoch {epoch+1}/{self.max_iter}", unit="batch", leave=False) as tepoch:
                for batch_idx, (inputs, targets) in enumerate(tepoch):
                    if self.cuda_:
                        inputs = inputs.cuda()
                        targets = targets.cuda()

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Compute loss based on loss_type
                    if self.loss_type == 'weighted_bce':
                        # Apply pos_weight manually in loss
                        weights = torch.where(targets == 1, pos_weight, 1.0)
                        if self.cuda_:
                            weights = weights.cuda()
                        loss = nn.functional.binary_cross_entropy(
                            outputs, targets.float(), weight=weights, reduction='mean')
                    else:
                        loss = self.criterion(outputs, targets.float())
                    
                    loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

    def predict_proba(self, X):
        X = X.to_numpy(dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # if self.cuda_:
        #     X_tensor = X_tensor.cuda()

        data_loader = torch.utils.data.DataLoader(
            X_tensor, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            all_outputs = []
            for batch_x in data_loader:
                if self.cuda_:
                    batch_x = batch_x.cuda()
                batch_outputs = self.model(batch_x)
                all_outputs.append(batch_outputs.cpu().numpy())
        outputs = np.concatenate(all_outputs)
        probs = np.stack([1 - outputs, outputs], 1)
        return probs


class MLPClassifier(NeuralClassifier):
    def __init__(
            self, hidden_layer_sizes,
            max_iter=50, weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
            dropout_p=0.0, use_residual=True, use_batch_norm=False,
            batch_size=512, shuffle=True, random_state=42, cuda=True, loss_type='weighted_bce') -> None:
        super().__init__(hidden_layer_sizes, max_iter, weight_decay, learning_rate, momentum, squares, optimizer_t,
                         dropout_p, use_residual, use_batch_norm, batch_size, shuffle, random_state, cuda, loss_type)

    def _build_model(self):
        self.model = MLP(self.input_size, self.hidden_layer_sizes, self.output_size,
                         self.dropout_p, self.use_residual, self.use_batch_norm)
        if self.cuda_:
            self.model = self.model.cuda()


class GNNClassifier(NeuralClassifier):
    def __init__(
            self, hidden_layer_sizes,
            max_iter=50, weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
            dropout_p=0.0, alpha=0.75, use_residual=True, use_batch_norm=False,
            batch_size=512, shuffle=True, random_state=42, cuda=False, loss_type='weighted_bce') -> None:
        super().__init__(hidden_layer_sizes, max_iter, weight_decay, learning_rate, momentum, squares, optimizer_t,
                         dropout_p, use_residual, use_batch_norm, batch_size, shuffle, random_state, cuda, loss_type)
        self.alpha = alpha

    def _build_model(self):
        self.model = GNN(self.input_size, self.hidden_layer_sizes, self.output_size,
                         self.dropout_p, self.alpha, self.use_residual, self.use_batch_norm)
        if self.cuda_:
            self.model = self.model.cuda()


class GCNClassifier(NeuralClassifier):
    def __init__(
            self, hidden_layer_sizes,
            max_iter=50, weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
            dropout_p=0.0, alpha=0.75, use_residual=True, use_batch_norm=False,
            batch_size=512, shuffle=True, random_state=42, cuda=False, loss_type='weighted_bce') -> None:
        super().__init__(hidden_layer_sizes, max_iter, weight_decay, learning_rate, momentum, squares, optimizer_t,
                         dropout_p, use_residual, use_batch_norm, batch_size, shuffle, random_state, cuda, loss_type)
        self.alpha = alpha

    def _build_model(self):
        self.model = GCN(self.input_size, self.hidden_layer_sizes, self.output_size,
                         self.dropout_p, self.alpha, self.use_residual, self.use_batch_norm)
        if self.cuda_:
            self.model = self.model.cuda()


class GATClassifier(NeuralClassifier):
    def __init__(
            self, hidden_layer_sizes,
            max_iter=50, weight_decay=1e-4, learning_rate=5e-3, momentum=0.9, squares=0.999, optimizer_t='sgdm',
            dropout_p=0.0, alpha=0.75, use_residual=True, use_batch_norm=False, heads=1,
            batch_size=512, shuffle=True, random_state=42, cuda=False, loss_type='weighted_bce') -> None:
        super().__init__(hidden_layer_sizes, max_iter, weight_decay, learning_rate, momentum, squares, optimizer_t,
                         dropout_p, use_residual, use_batch_norm, batch_size, shuffle, random_state, cuda, loss_type)
        self.alpha = alpha
        self.heads = heads

    def _build_model(self):
        self.model = GAT(self.input_size, self.hidden_layer_sizes, self.output_size,
                         self.dropout_p, self.alpha, self.use_residual, self.use_batch_norm, heads=self.heads)
        if self.cuda_:
            self.model = self.model.cuda()


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    def calculate_metrics(labels, probs):
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
        preds_categorical = np.where(
            probs > 0.5, 1.0, 0.0)
        acc = np.mean(preds_categorical == labels)
        return np.array([auc, aupr, acc])

    def print_metrics(auc, aupr, acc, valid=False, internal=True):
        if valid:
            print(f"Valid:         ", end='')
        else:
            if internal:
                print(f"Internal test: ", end='')
            else:
                print(f"External test: ", end='')
        print(
            f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | Acc: {acc * 100:.2f}%")

    n_samples, n_features = 32, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples).astype(np.float64)
    mlp_classifier = MLPClassifier(max_iter=1, cuda=False)
    mlp_classifier.fit(X, y)
    probs = mlp_classifier.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y, probs)
    print_metrics(*metrics)
