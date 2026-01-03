import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
except ImportError:
    print("Aviso: LightGBM não instalado. O modelo será ignorado na comparação.")
    has_lgbm = False

FILE_PATH = 'Obesity_normalizado_ok.csv'
TARGET_COLUMN = 'Obesity'
RANDOM_SEED = 42


class BMICalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Evitar divisão por zero para Height
        X_copy['Height'] = X_copy['Height'].replace(0, np.nan) # Substituir 0 por NaN
        # Preencher NaN em Height com a mediana antes do cálculo do BMI
        X_copy['Height'] = X_copy['Height'].fillna(X_copy['Height'].median())

        # Calcular BMI
        X_copy['BMI'] = X_copy['Weight'] / (X_copy['Height'] ** 2)
        return X_copy


def load_and_clean_data(file_path):

    try:
        df = pd.read_csv(file_path, sep=',')

        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=';')

        original_cols = [col for col in df.columns if not col.startswith('N_') and col != TARGET_COLUMN]
        df = df[original_cols + [TARGET_COLUMN]]

        cols_to_convert = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Tratamento de dados ausentes (imputação simples)
        for col in cols_to_convert:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        categorical_features_clean = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in categorical_features_clean:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
                df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.strip()

        return df

    except Exception as e:
        print(f"Erro ao carregar ou limpar o CSV: {e}")
        return pd.DataFrame()

df = load_and_clean_data(FILE_PATH)

if df.empty or TARGET_COLUMN not in df.columns:
    print("Falha crítica no carregamento de dados. Não é possível continuar o treinamento.")
    exit()

# Definição final de X e y
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Codificação da variável alvo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'label_encoder.pkl')

# Divisão dos dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded)
print(f"Dataset de Treino: {len(X_train)} amostras | Dataset de Teste: {len(X_test)} amostras")


# Definição das colunas
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI'] # Adicionado 'BMI' aqui

# Criando o pré-processador:
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

try:
    preprocessor.set_output(transform="pandas")
except AttributeError:
    print("Warning: `set_output(transform='pandas')` is not available in this scikit-learn version. The feature name warning might persist.")


classifiers = {
    'Logistic Regression (Baseline)': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, random_state=RANDOM_SEED, class_weight='balanced'),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_SEED),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_SEED),
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_SEED, estimator=DecisionTreeClassifier(max_depth=1)),
    'SVM (RBF Kernel)': SVC(random_state=RANDOM_SEED, class_weight='balanced', probability=True),
    'Naive Bayes (Gaussian)': GaussianNB()
}

#if has_lgbm:
#    classifiers['LightGBM'] = LGBMClassifier(random_state=RANDOM_SEED, class_weight='balanced')


results = {}
best_model_name = ''
best_accuracy = 0.0

print("\n--- INICIANDO TREINAMENTO E COMPARAÇÃO DE 10 MODELOS ---")

for name, classifier in classifiers.items():
    print(f"\n-> Treinando Pipeline: {name}...")

    # Construção do Pipeline para o modelo atual
    model_pipeline_current = Pipeline(steps=[
        ('bmi_calc', BMICalculator()),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Treinamento
    model_pipeline_current.fit(X_train, y_train)

    # Avaliação
    y_pred = model_pipeline_current.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'pipeline': model_pipeline_current
    }

    # Imprimindo resultados
    print(f"   Acurácia: {accuracy * 100:.2f}%")
    print(f"   Relatório de Classificação:\n{classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)}")
    # Adiciona a Matriz de Confusão
    print(f"   Matriz de Confusão:\n{confusion_matrix(y_test, y_pred)}")

    # Identificando o melhor modelo
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name

print("\n-----------------------------------------------------------")
print(f"COMPARAÇÃO FINAL: O MELHOR MODELO FOI o {best_model_name} com Acurácia de {best_accuracy * 100:.2f}%")
print(f"Total de Modelos Comparados: {len(classifiers)}")
print("-----------------------------------------------------------")


final_model_pipeline = results[best_model_name]['pipeline']
joblib.dump(final_model_pipeline, 'obesity_prediction_model_pipeline.pkl')
print(f"\nPipeline do melhor modelo ({best_model_name}) salvo como 'obesity_prediction_model_pipeline.pkl'.")