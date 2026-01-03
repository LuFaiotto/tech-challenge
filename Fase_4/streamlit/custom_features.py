# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BMICalculator(BaseEstimator, TransformerMixin):
    """Calcula o IMC (BMI) e substitui as colunas Peso e Altura originais."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Cálculo do IMC: Peso (kg) / Altura (m)^2
        # As colunas Height e Weight devem ser numéricas.
        X_copy['BMI'] = X_copy['Weight'] / (X_copy['Height'] ** 2)
        # Remove Height e Weight originais para usar o BMI como feature principal
        return X_copy.drop(columns=['Height', 'Weight'])
