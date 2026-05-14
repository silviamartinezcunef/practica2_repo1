"""
ModelWrapper para hacer compatible LightGBM/XGBoost con sklearn's CalibratedClassifierCV
"""
import numpy as np
import xgboost as xgb


class ModelWrapper:
    """
    Wrapper para modelos LightGBM/XGBoost que los hace compatibles
    con sklearn's CalibratedClassifierCV.
    """

    def __init__(self, model, model_type):
        """
        Args:
            model: Modelo de LightGBM o XGBoost entrenado
            model_type: 'lightgbm' o 'xgboost'
        """
        self.model = model
        self.model_type = model_type
        self.classes_ = np.array([0, 1])  # Clases binarias

    def predict_proba(self, X):
        """
        Predice probabilidades para ambas clases.

        Args:
            X: Features (DataFrame o array)

        Returns:
            Array de shape (n_samples, 2) con probabilidades [P(clase=0), P(clase=1)]
        """
        if self.model_type == 'lightgbm':
            proba = self.model.predict(X)
        else:  # xgboost
            dmatrix = xgb.DMatrix(X)
            proba = self.model.predict(dmatrix)

        # Devolver array 2D: [P(clase=0), P(clase=1)]
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        """
        Predice la clase (0 o 1).

        Args:
            X: Features (DataFrame o array)

        Returns:
            Array de shape (n_samples,) con predicciones de clase
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
