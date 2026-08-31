"""Petit réseau de neurones tabulaire (Keras 3, backend PyTorch).

On garde le même cadre que les autres modèles : mêmes features (sans fuite),
même découpage train/test, même pré-traitement `scikit-learn` en amont. Le
réseau est volontairement simple — l'objectif est de comparer honnêtement le
deep learning aux modèles classiques sur ce problème, pas de sur-optimiser.

Le backend Keras doit être choisi AVANT l'import de keras :
``os.environ["KERAS_BACKEND"] = "torch"``.
"""
from __future__ import annotations

import numpy as np


def build_mlp(n_features: int, *, largeur: tuple[int, ...] = (64, 32),
              dropout: float = 0.3, lr: float = 1e-3):
    """MLP binaire : quelques couches denses, dropout, sortie sigmoïde.

    Compilé avec l'entropie croisée binaire et suivi de l'AUC-PR (metric la plus
    parlante vu le déséquilibre).
    """
    import keras

    couches = [keras.layers.Input((n_features,))]
    for u in largeur:
        couches += [keras.layers.Dense(u, activation="relu"),
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(dropout)]
    couches += [keras.layers.Dense(1, activation="sigmoid")]

    model = keras.Sequential(couches)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="ap")],
    )
    return model


def train_mlp(model, X_tr, y_tr, X_val, y_val, *, epochs: int = 120,
              batch_size: int = 64, patience: int = 15):
    """Entraîne avec arrêt anticipé sur l'AUC-PR de validation et pondération
    des classes (déséquilibre ~1/3)."""
    import keras

    y_tr = np.asarray(y_tr)
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}

    hist = model.fit(
        np.asarray(X_tr, dtype="float32"), y_tr.astype("float32"),
        validation_data=(np.asarray(X_val, dtype="float32"), np.asarray(y_val, dtype="float32")),
        epochs=epochs, batch_size=batch_size, verbose=0, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_ap", mode="max",
                                                 patience=patience, restore_best_weights=True)],
    )
    return hist


def predict_proba(model, X) -> np.ndarray:
    return model.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
