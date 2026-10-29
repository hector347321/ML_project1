import os
import time
import itertools
import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import sigmoid

DATA_PATH = "data/"
OUTPUT_FILE = "submission.csv"
TUNE = True             # recherche d'hyperparametres par CV
ABLATION = False         # comparation des variantes de preprocessing par CV 
QUICK = False            # True = sous-echantillon (1 ligne sur 50) pour deboguer
SEED = 42
MAX_ITERS = 400

# Options de preprocessing (comparees par ABLATION = True)
NAN_INDICATORS = True    
NAN_IND_MIN_FRAC = 0.05 
MAX_LEVELS = 15         

# Meilleures valeurs trouvees par CV
BEST_LAMBDA = 0.001        
BEST_POS_WEIGHT = 2.0    
BEST_THRESHOLD = 0.3325     

# Grille de recherche (utilisee seulement si TUNE = True)
K_FOLDS = 4
GRID_LAMBDA = [1e-5, 1e-4, 1e-3]
GRID_POS_WEIGHT = [2.0, 3.0, 4.0, 6.0]   # poids des positifs dans la loss
THRESHOLDS = np.linspace(0.05, 0.95, 91)

# Variantes comparees par ABLATION = True (avec BEST_LAMBDA et BEST_POS_WEIGHT)
ABLATION_VARIANTS = [
    ("baseline (one-hot <= 6, sans indicateurs)", {"nan_indicators": False, "max_levels": 6}),
    ("+ indicateurs de manquants",                {"nan_indicators": True,  "max_levels": 6}),
    ("+ one-hot etendu (<= 15)",                  {"nan_indicators": False, "max_levels": 15}),
    ("+ les deux",                                {"nan_indicators": True,  "max_levels": 15}),
]

def load_feature_names(data_path):
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return np.array(header[1:])


def fit_special_codes(X):
    """Decide pour chaque colonne quels codes deviennent NaN et lesquels deviennent 0.

    Returns:
        rules: dict {colonne: (codes -> NaN, codes -> 0)}
    """
    rules = {}
    for j in range(X.shape[1]):
        col = X[:, j]
        vals = np.unique(col[~np.isnan(col)])
        if len(vals) == 0:
            continue
        mx = vals.max()
        nuniq = len(vals)

        # "nombre de jours" : 88 = aucun, 77 = ne sait pas, 99 = refus
        if 88 in vals and 77 in vals and 99 in vals:
            rules[j] = ([77, 99], [88])
        elif mx == 9 and nuniq <= 15:
            if 8 in vals:
                rules[j] = ([9], [])
            else:
                rules[j] = ([7, 9], [])
        elif mx == 99 and nuniq <= 15:
            rules[j] = ([77, 99], [])
        elif mx == 999 and 777 in vals:
            rules[j] = ([777, 999], [])
    return rules


def apply_special_codes(X, rules):
    """Applique les regles apprises sur le train a n'importe quel tableau."""
    X = X.copy()
    for j, (to_nan, to_zero) in rules.items():
        col = X[:, j]
        col[np.isin(col, to_zero)] = 0
        col[np.isin(col, to_nan)] = np.nan
    return X

def fit_preprocessing(X, nan_threshold=0.8, one_hot=True, max_levels=MAX_LEVELS,
                      min_freq=0.01, nan_indicators=NAN_INDICATORS):
    """Apprend sur X (deja nettoye) : colonnes gardees, medianes, moyennes,
    ecarts-types, modalites one-hot, colonnes recevant un indicateur de manquant."""
    nan_frac = np.mean(np.isnan(X), axis=0)
    with np.errstate(invalid="ignore"):
        std_raw = np.nanstd(X, axis=0)
    keep = np.where((nan_frac < nan_threshold) & (std_raw > 0))[0]
    X = X[:, keep]

    # colonnes (parmi les gardees) qui recoivent un indicateur 0/1 "etait manquant"
    if nan_indicators:
        nan_cols = np.where(nan_frac[keep] >= NAN_IND_MIN_FRAC)[0]
    else:
        nan_cols = np.array([], dtype=int)

    median = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), median, X)
    mean = X_imp.mean(axis=0)
    std = X_imp.std(axis=0)
    std[std == 0] = 1.0

    # modalites one-hot pour les colonnes a peu de valeurs (3 a max_levels)
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
                    levels[j] = lv[1:] 

    return {"keep": keep, "median": median, "mean": mean, "std": std, "levels": levels,
            "nan_cols": nan_cols}


def transform(X, p):
    """Applique un preprocessing appris par fit_preprocessing.
    Ordre des colonnes : biais | numeriques standardisees | one-hot | indicateurs."""
    X = X[:, p["keep"]]
    blocks = [(X[:, [j]] == lv[None, :]).astype(np.float32) for j, lv in p["levels"].items()]
    nan_ind = np.isnan(X[:, p["nan_cols"]]).astype(np.float32)
    Xn = np.where(np.isnan(X), p["median"], X)
    Xn = ((Xn - p["mean"]) / p["std"]).astype(np.float32)
    bias = np.ones((X.shape[0], 1), dtype=np.float32)
    return np.hstack([bias, Xn] + blocks + [nan_ind])


def fit_transform_train_test(X_tr_raw, X_te_raw, **prep_opts):
    """Regles de nettoyage + preprocessing appris sur X_tr_raw,
    appliques a X_tr_raw et X_te_raw. prep_opts : options de fit_preprocessing."""
    rules = fit_special_codes(X_tr_raw)
    X_tr = apply_special_codes(X_tr_raw, rules)
    X_te = apply_special_codes(X_te_raw, rules)
    params = fit_preprocessing(X_tr, **prep_opts)
    return transform(X_tr, params), transform(X_te, params), params


# ----------------------------------------------------------------------------
# 4. MODELE : regression logistique regularisee L2 avec poids de classe,
#    descente de gradient acceleree (Nesterov)
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
        z = w + (k - 1) / (k + 2) * (w - w_prev)  # pas de Nesterov
        grad = tx.T @ (sw * (sigmoid(tx @ z) - y01)) / N
        reg = 2 * lambda_ * z
        reg[0] = 0.0  # biais non regularise
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


def cross_validate(y01, X_raw, lambda_, pos_weight, k=K_FOLDS, **prep_opts):
    """CV stratifiee. Le nettoyage et le preprocessing sont refaits dans chaque fold,
    appris sur les folds d'entrainement uniquement : pas de fuite vers la validation."""
    folds = stratified_folds(y01, k, SEED)
    f1s, thrs, accs = [], [], []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        tx_tr, tx_val, _ = fit_transform_train_test(X_raw[tr_idx], X_raw[val_idx], **prep_opts)
        w = weighted_reg_logistic(y01[tr_idx], tx_tr, lambda_, pos_weight)
        probs = predict_proba(tx_val, w)
        thr, f1 = best_threshold(y01[val_idx], probs)
        f1s.append(f1)
        thrs.append(thr)
        accs.append(accuracy(y01[val_idx], (probs >= thr).astype(int)))
    return np.mean(f1s), np.std(f1s), np.mean(thrs), np.mean(accs)


def tune(y01, X_raw):
    """Grid search sur (lambda, pos_weight), chaque couple evalue par CV."""
    results = []
    print("\nlambda    pos_w |   F1 (moy ± std)  | seuil | accuracy")
    for lam, pw in itertools.product(GRID_LAMBDA, GRID_POS_WEIGHT):
        f1, f1_std, thr, acc = cross_validate(y01, X_raw, lam, pw)
        results.append((f1, lam, pw, thr))
        print(f"{lam:<8g} {pw:5.1f} | {f1:.4f} ± {f1_std:.4f} | {thr:.2f}  | {acc:.4f}")

    best_f1, best_lam, best_pw, best_thr = max(results)
    print(f"\nMeilleur F1 en CV : {best_f1:.4f}")
    print("A recopier dans la CONFIG pour regenerer la soumission avec TUNE = False :")
    print(f"BEST_LAMBDA = {best_lam}")
    print(f"BEST_POS_WEIGHT = {best_pw}")
    print(f"BEST_THRESHOLD = {best_thr:.4f}")
    return best_lam, best_pw, best_thr


def ablation(y01, X_raw):
    """Compare les variantes de preprocessing par CV, a hyperparametres fixes
    (BEST_LAMBDA, BEST_POS_WEIGHT). Le tableau obtenu sert directement au rapport."""
    print(f"\nAblation (lambda={BEST_LAMBDA}, pos_weight={BEST_POS_WEIGHT}, {K_FOLDS} folds)")
    print(f"{'variante':<44}|   F1 (moy ± std)  | seuil | accuracy")
    for name, opts in ABLATION_VARIANTS:
        f1, f1_std, thr, acc = cross_validate(y01, X_raw, BEST_LAMBDA, BEST_POS_WEIGHT, **opts)
        print(f"{name:<44}| {f1:.4f} ± {f1_std:.4f} | {thr:.2f}  | {acc:.4f}")


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

    # --- comparaison des variantes de preprocessing (pas de soumission)
    if ABLATION:
        ablation(y01, x_train)
        print(f"\nTemps total : {time.time() - t0:.0f}s")
        raise SystemExit

    # --- hyperparametres : recherche par CV, ou valeurs deja trouvees
    if TUNE:
        lambda_, pos_weight, threshold = tune(y01, x_train)
    else:
        lambda_, pos_weight, threshold = BEST_LAMBDA, BEST_POS_WEIGHT, BEST_THRESHOLD
    print(f"\nHyperparametres utilises : lambda={lambda_}, pos_weight={pos_weight}, "
          f"seuil={threshold:.4f}")

    # --- nettoyage + preprocessing (appris sur le train, applique au train et au test)
    tx_train, tx_test, params = fit_transform_train_test(x_train, x_test)
    n_onehot = sum(len(lv) for lv in params["levels"].values())
    print(f"Apres preprocessing : train {tx_train.shape} | test {tx_test.shape} "
          f"| colonnes gardees : {len(params['keep'])} | one-hot : {n_onehot} "
          f"| indicateurs de manquants : {len(params['nan_cols'])}")
    assert tx_train.shape[1] == tx_test.shape[1]
    assert not np.isnan(tx_train).any() and not np.isnan(tx_test).any()

    # --- entrainement final sur tout le train
    w = weighted_reg_logistic(y01, tx_train, lambda_, pos_weight)

    # features les plus importantes (colonnes numeriques standardisees uniquement)
    n_num = len(params["keep"])
    imp = np.abs(w[1:1 + n_num])
    top = np.argsort(-imp)[:10]
    print("\nTop 10 features (|poids|) :")
    for j in top:
        print(f"  {feature_names[params['keep'][j]]:<15} {w[1 + j]:+.3f}")

    # --- prediction test
    probs_test = predict_proba(tx_test, w)
    y_pred = np.where(probs_test >= threshold, 1, -1)
    print(f"\nPart de +1 predits sur le test : {np.mean(y_pred == 1):.3f}")

    create_csv_submission(test_ids, y_pred, OUTPUT_FILE)
    print(f"Soumission ecrite : {OUTPUT_FILE} ({len(y_pred)} lignes) "
          f"en {time.time() - t0:.0f}s")
