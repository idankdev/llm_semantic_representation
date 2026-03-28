"""
Example:
  python represent.py \
    --train-csv /path/to/datasets/mydataset/train.csv \
    --test-csv /path/to/datasets/mydataset/test.csv \
    --question-embeddings /path/to/question_embeddings.pth \
    --lmbda 1.0 --eps 0.1 --save-em /tmp/EM.pth --run-router

Output:
  Prints accuracy and AUC. Optionally saves model embeddings (E_M) and
  runs a simple router evaluation (nearest-model by dot product).
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc


def tikhonov_regularized_pseudoinverse(A: torch.Tensor, lmbda: float = 1.0, eps: float = 1e-1) -> torch.Tensor:
    """Compute a numerically-stable Tikhonov-regularized pseudoinverse of A.

    A: shape (m, n)
    returns: A_pinv shape (n, m)
    """
    # Use SVD for stability
    U, S, Vh = torch.svd(A)
    # threshold small singular values
    S_threshold = S.max() * torch.finfo(S.dtype).eps * A.shape[0]
    S = torch.where(S > S_threshold, S, torch.zeros_like(S))
    # further threshold with eps
    S = torch.where(S > eps, S, torch.zeros_like(S))
    S2 = S ** 2
    D = S / (S2 + 2.0 * lmbda)
    Dmat = torch.diag(D)
    A_pinv = Vh @ Dmat @ U.t()
    return A_pinv


def run_pi(train_csv: str, test_csv: str, question_embeddings_file: str, lmbda: float = 1.0, eps: float = 0.1,
           device: str = 'cpu', save_em: str = None):
    """Run success prediction on a single dataset and embedding file.

    Returns (y_true_all, y_pred_all, E_M (tensor), val_df (DataFrame)).
    """
    device = torch.device(device)

    train_df = pd.read_csv(train_csv, index_col='prompt_id')
    test_df = pd.read_csv(test_csv, index_col='prompt_id')

    # load embeddings (torch saved tensor of shape (num_prompts, dim))
    embeddings = torch.load(question_embeddings_file, map_location=device, weights_only=True)

    # select embeddings rows by prompt order
    train_embeddings = embeddings[train_df.index.values].to(device)  # (train_size, dim)
    val_embeddings = embeddings[test_df.index.values].to(device)    # (val_size, dim)

    model_ids = train_df.columns
    # train_perf: (num_models, train_size)
    train_perf = torch.tensor(train_df[model_ids].T.values, device=device, dtype=torch.float32)
    # val_perf as (num_models, val_size) so flattening order matches y_pred
    val_perf = test_df[model_ids].T  # (num_models, val_size)

    # replace zeros with -1 to follow repo convention (0 -> -1)
    train_perf[train_perf == 0.0] = -1.0
    val_perf = val_perf.copy()
    val_perf[val_perf == 0.0] = -1.0

    # compute pseudoinverse of train embeddings transpose (shape => train_size x dim)
    # input to pseudoinverse should be (dim, train_size)
    A = train_embeddings.t()  # (dim, train_size)
    A_pinv = tikhonov_regularized_pseudoinverse(A, lmbda=lmbda, eps=eps)  # (train_size, dim)

    # compute model embedding matrix E_M: (num_models, dim)
    E_M = train_perf @ A_pinv  # (num_models, dim)

    # predict on validation: y_pred = E_M @ val_embeddings.T -> (num_models, val_size)
    y_pred = (E_M @ val_embeddings.t()).cpu().numpy()

    # flatten ground truth and predictions: both are (num_models, val_size)
    y_true_all = val_perf.values.flatten()
    y_pred_all = y_pred.flatten()

    # optionally save E_M
    if save_em is not None:
        torch.save(E_M.cpu(), save_em)

    return y_true_all, y_pred_all, E_M.cpu(), val_perf


def compute_metrics(y_true_all: np.ndarray, y_pred_all: np.ndarray):
    """Compute accuracy (sign-based) and AUC.

    The repo uses -1/1 labels for incorrect/correct; compute accuracy
    by sign agreement and AUC using binary labels.
    """
    # accuracy: sign comparison
    y_true_sign = np.sign(y_true_all)
    y_pred_sign = np.sign(y_pred_all)
    acc = np.mean(y_true_sign == y_pred_sign)

    # AUC: need binary labels 0/1
    y_true_bin = (y_true_all > 0).astype(int)
    try:
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred_all)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = float('nan')

    return {'accuracy': float(acc), 'auc': float(roc_auc)}


def run_router(E_M: torch.Tensor, question_embeddings_file: str, test_df: pd.DataFrame, device: str = 'cpu'):
    """
    Run model selection evaluation on the given test set.

    Returns (accuracy, recall).
    """
    device = torch.device(device)
    E_M = E_M.to(device)  # (num_models, dim)

    question_embeddings = torch.load(question_embeddings_file, map_location=device, weights_only=True).to(device)
    test_df = test_df.T # (test_size, num_models)
    test_df[test_df == -1] = 0  # convert back to 0/1 for labels
    test_question_embeddings = question_embeddings[test_df.index.values]  # (test_size, dim)
    test_labels = test_df.values  # (test_size, num_models)
    similarities = (test_question_embeddings @ E_M.T).cpu().numpy() # (test_size, num_models)
    I = similarities.argmax(axis=1).reshape(-1, 1) # (test_size, 1)
    y_pred = np.take_along_axis(test_labels, I, axis=1).squeeze() # (test_size,)
    accuracy = y_pred.mean()
    recall = y_pred.sum() / test_labels.max(axis=1).sum()
    return accuracy, recall

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-csv', required=True)
    p.add_argument('--test-csv', required=True)
    p.add_argument('--question-embeddings', required=True, help='torch .pth file with question embeddings')
    p.add_argument('--lmbda', type=float, default=1.0)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--save-em', default=None, help='path to save E_M (torch .pth)')
    p.add_argument('--run-router', action='store_true', help='run simple router evaluation')
    args = p.parse_args()

    y_true_all, y_pred_all, E_M, val_df = run_pi(args.train_csv, args.test_csv, args.question_embeddings,
                                                lmbda=args.lmbda, eps=args.eps, device=args.device, save_em=args.save_em)

    metrics = compute_metrics(y_true_all, y_pred_all)
    print('Success Prediction Results:')
    print(f"[*]\tAccuracy (sign): {metrics['accuracy']:.4f}")
    print(f"[*]\tAUC: {metrics['auc']:.4f}")

    if args.run_router:
        acc, recall = run_router(E_M, args.question_embeddings, val_df, device=args.device)
        print('Model Selection Results:')
        print(f"[*]\tAccuracy: {acc:.4f}")
        print(f"[*]\tRecall: {recall:.4f}")


if __name__ == '__main__':
    main()
