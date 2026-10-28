"""
Pipeline complet BRFSS / MICHD
------------------------------
Fichiers attendus dans le meme dossier (ou adapter les imports) :
  - implementations.py : vos 6 fonctions + sigmoid
  - helpers.py         : load_csv_data (et create_csv_submission si vous l'avez)
Lancer :  python run.py
"""
import os
import time
import itertools
import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import sigmoid

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = "data/"
OUTPUT_FILE = "submission.csv"
QUICK = False            # True = sous-echantillon (1 ligne sur 50) pour deboguer
SEED = 42
K_FOLDS = 4
MAX_ITERS = 400
GRID_LAMBDA = [1e-5, 1e-4, 1e-3]
GRID_POS_WEIGHT = [1.0, 4.0]      # poids des positifs dans la loss
THRESHOLDS = np.linspace(0.05, 0.95, 91)


# ----------------------------------------------------------------------------
# 1. CHARGEMENT
# ----------------------------------------------------------------------------
def load_feature_names(data_path):
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return np.array(header[1:])


# ----------------------------------------------------------------------------
# 2. NETTOYAGE DES CODES SPECIAUX (heuristique generique)
#    A raffiner avec le codebook BRFSS 2015 colonne par colonne.
# ----------------------------------------------------------------------------
def clean_special_codes(X):
    """7/9, 77/99, 777/999 = 'ne sait pas' / 'refus' -> NaN ; 88 = 'aucun' -> 0."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        vals = np.unique(col[~np.isnan(col)])
        if len(vals) == 0:
            continue
        mx = vals.max()
        nuniq = len(vals)

        # colonnes de type "nombre de jours" : 88 = aucun, 77 = ne sait pas, 99 = refus
        if 88 in vals and 77 in vals and 99 in vals:
            col[col == 88] = 0
            col[(col == 77) | (col == 99)] = np.nan
        # colonnes categorielles a 1 chiffre
        elif mx == 9 and nuniq <= 15:
            col[(col == 7) | (col == 9)] = np.nan
        # categorielles a 2 chiffres
        elif mx == 99 and nuniq <= 15:
            col[(col == 77) | (col == 99)] = np.nan
        # codes a 3 chiffres
        elif mx == 999 and 777 in vals:
            col[(col == 777) | (col == 999)] = np.nan
        X[:, j] = col
    return X


# ----------------------------------------------------------------------------
# 3. PREPROCESSING (fit sur le train uniquement, transform sur train et test)
# ----------------------------------------------------------------------------
def fit_preprocessing(X, nan_threshold=0.8, one_hot=True, max_levels=6, min_freq=0.01):
    nan_frac = np.mean(np.isnan(X), axis=0)
    std_raw = np.nanstd(X, axis=0)
    keep = np.where((nan_frac < nan_threshold) & (std_raw > 0))[0]
    X = X[:, keep]

    median = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), median, X)
    mean = X_imp.mean(axis=0)
    std = X_imp.std(axis=0)
    std[std == 0] = 1.0

    # niveaux one-hot pour les colonnes a peu de modalites (3 a max_levels)
    levels = {}
    if one_hot:
        n = X.shape[0]
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            vals, counts = np.unique(col, return_counts=True)
            if 3 <= len(vals) <= max_levels:
                lv = vals[counts / n >= min_freq]
                if len(lv) >= 2:
                    levels[j] = lv[1:]          # on retire la 1re modalite (reference)

    return {"keep": keep, "median": median, "mean": mean, "std": std, "levels": levels}


def transform(X, p):
    X = X[:, p["keep"]]
    blocks = [(X[:, [j]] == lv[None, :]).astype(np.float32) for j, lv in p["levels"].items()]
    Xn = np.where(np.isnan(X), p["median"], X)
    Xn = ((Xn - p["mean"]) / p["std"]).astype(np.float32)
    bias = np.ones((X.shape[0], 1), dtype=np.float32)
    return np.hstack([bias, Xn] + blocks)


# ----------------------------------------------------------------------------
# 4. MODELE : regression logistique regularisee L2 avec poids de classe
#    (= reg_logistic_regression + poids), descente de gradient acceleree (Nesterov)
# ----------------------------------------------------------------------------
def lipschitz(tx, sw, lambda_, n_iter=30):
    """Estime la constante de Lipschitz du gradient par iteration de puissance."""
    rng = np.random.RandomState(0)
    v = rng.randn(tx.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(n_iter):
        u = tx.T @ (sw * (tx @ v)) / tx.shape[0]
        lam = np.linalg.norm(u)
        v = u / lam
    return 0.25 * lam + 2 * lambda_


def weighted_reg_logistic(y01, tx, lambda_, pos_weight, max_iters=MAX_ITERS):
    y01 = y01.astype(np.float32)
    sw = np.where(y01 == 1, pos_weight, 1.0).astype(np.float32)
    sw /= sw.mean()
    gamma = 1.0 / lipschitz(tx, sw, lambda_)
    N = tx.shape[0]

    w = np.zeros(tx.shape[1], dtype=np.float32)
    w_prev = w.copy()
    for k in range(1, max_iters + 1):
        z = w + (k - 1) / (k + 2) * (w - w_prev)          # pas de Nesterov
        grad = tx.T @ (sw * (sigmoid(tx @ z) - y01)) / N
        reg = 2 * lambda_ * z
        reg[0] = 0.0                                       # biais non regularise
        w_prev = w
        w = (z - gamma * (grad + reg)).astype(np.float32)
    return w


def predict_proba(tx, w):
    return sigmoid(tx @ w)


# ----------------------------------------------------------------------------
# 5. METRIQUES + VALIDATION CROISEE STRATIFIEE
# ----------------------------------------------------------------------------
def f1_score(y_true01, y_pred01):
    tp = np.sum((y_pred01 == 1) & (y_true01 == 1))
    fp = np.sum((y_pred01 == 1) & (y_true01 == 0))
    fn = np.sum((y_pred01 == 0) & (y_true01 == 1))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def accuracy(y_true01, y_pred01):
    return float(np.mean(y_true01 == y_pred01))


def best_threshold(y_true01, probs):
    scores = [f1_score(y_true01, (probs >= t).astype(int)) for t in THRESHOLDS]
    i = int(np.argmax(scores))
    return THRESHOLDS[i], scores[i]


def stratified_folds(y01, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx = rng.permutation(np.where(y01 == c)[0])
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].append(part)
    return [np.concatenate(f) for f in folds]


def cross_validate(y01, tx, lambda_, pos_weight, k=K_FOLDS):
    folds = stratified_folds(y01, k, SEED)
    f1s, thrs, accs = [], [], []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        w = weighted_reg_logistic(y01[tr_idx], tx[tr_idx], lambda_, pos_weight)
        probs = predict_proba(tx[val_idx], w)
        thr, f1 = best_threshold(y01[val_idx], probs)
        f1s.append(f1)
        thrs.append(thr)
        accs.append(accuracy(y01[val_idx], (probs >= thr).astype(int)))
    return np.mean(f1s), np.std(f1s), np.mean(thrs), np.mean(accs)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    # --- chargement
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH, sub_sample=QUICK)
    feature_names = load_feature_names(DATA_PATH)
    y01 = ((y_train + 1) / 2).astype(int)
    print(f"x_train {x_train.shape} | x_test {x_test.shape} | positifs : {y01.mean():.4f}")

    # --- nettoyage + preprocessing
    x_train = clean_special_codes(x_train)
    x_test = clean_special_codes(x_test)
    params = fit_preprocessing(x_train)
    tx_train = transform(x_train, params)
    tx_test = transform(x_test, params)
    print(f"Apres preprocessing : train {tx_train.shape} | test {tx_test.shape} "
          f"| colonnes gardees : {len(params['keep'])} | colonnes one-hot : "
          f"{tx_train.shape[1] - 1 - len(params['keep'])}")
    assert tx_train.shape[1] == tx_test.shape[1]
    assert not np.isnan(tx_train).any() and not np.isnan(tx_test).any()

    # --- recherche d'hyperparametres par CV
    results = []
    print("\nlambda    pos_w |   F1 (moy ± std)  | seuil | accuracy")
    for lam, pw in itertools.product(GRID_LAMBDA, GRID_POS_WEIGHT):
        f1, f1_std, thr, acc = cross_validate(y01, tx_train, lam, pw)
        results.append((f1, lam, pw, thr))
        print(f"{lam:<8g} {pw:5.1f} | {f1:.4f} ± {f1_std:.4f} | {thr:.2f}  | {acc:.4f}")

    best_f1, best_lam, best_pw, best_thr = max(results)
    print(f"\nMeilleur : lambda={best_lam}, pos_weight={best_pw}, "
          f"seuil={best_thr:.2f}, F1 CV={best_f1:.4f}")

    # --- entrainement final sur tout le train
    w = weighted_reg_logistic(y01, tx_train, best_lam, best_pw)

    # features les plus importantes (colonnes numeriques standardisees uniquement)
    n_num = len(params["keep"])
    imp = np.abs(w[1:1 + n_num])
    top = np.argsort(-imp)[:10]
    print("\nTop 10 features (|poids|) :")
    for j in top:
        print(f"  {feature_names[params['keep'][j]]:<15} {w[1 + j]:+.3f}")

    # --- prediction test
    probs_test = predict_proba(tx_test, w)
    y_pred = np.where(probs_test >= best_thr, 1, -1)
    print(f"\nPart de +1 predits sur le test : {np.mean(y_pred == 1):.3f}")

    create_csv_submission(test_ids, y_pred, OUTPUT_FILE)
    print(f"Soumission ecrite : {OUTPUT_FILE} ({len(y_pred)} lignes) "
          f"en {time.time() - t0:.0f}s")
