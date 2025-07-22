import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path)

    # Criação do alvo
    y = ((df['QT_MAT_BAS_18_MAIS'] / (df['QT_MAT_BAS'] + 1)) > 0.15).astype(int)

    # Remover variáveis diretamente relacionadas ao alvo
    X = df.drop(columns=['QT_MAT_BAS_18_MAIS', 'QT_MAT_BAS'])

    return X, y

def plot_cv_results(cv_results):
    """Plot cross-validation results with mean and standard deviation"""
    
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [cv_results[f'test_{metric}'].mean() for metric in metrics]
    stds = [cv_results[f'test_{metric}'].std() for metric in metrics]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart with error bars
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    bars = ax1.bar(metrics, means, yerr=stds, capsize=10, color=colors, alpha=0.8, 
                   error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Resultados da Validação Cruzada - Resumo', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot showing distribution across folds
    cv_data = []
    for metric in metrics:
        for fold_score in cv_results[f'test_{metric}']:
            cv_data.append({'Métrica': metric.capitalize(), 'Score': fold_score})
    
    cv_df = pd.DataFrame(cv_data)
    
    sns.boxplot(data=cv_df, x='Métrica', y='Score', ax=ax2, palette=colors)
    ax2.set_title('Distribuição dos Scores por Fold', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add individual points (strip plot)
    sns.stripplot(data=cv_df, x='Métrica', y='Score', ax=ax2, color='black', 
                  alpha=0.6, size=6)
    
    plt.tight_layout()
    plt.savefig('models/cv_results_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed fold-by-fold plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create fold-by-fold comparison
    n_folds = len(cv_results['test_accuracy'])
    x = np.arange(n_folds)
    width = 0.2
    
    for i, metric in enumerate(metrics):
        scores = cv_results[f'test_{metric}']
        ax.bar(x + i*width, scores, width, label=metric.capitalize(), 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance por Fold - Validação Cruzada', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('models/cv_results_folds.png', dpi=300, bbox_inches='tight')
    plt.show()


from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score

def train_model_cv(X, y, cv_folds=3):
    # Identificar colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pré-processadores
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Modelo
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])

    # Métricas personalizadas
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0)
    }

    print(f"[INFO] Realizando cross-validation com {cv_folds} folds...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)

    print("\n[INFO] Resultados da Validação Cruzada (Média ± Desvio Padrão):")
    for metric in scoring.keys():
        mean = results[f'test_{metric}'].mean()
        std = results[f'test_{metric}'].std()
        print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")

    # Plot cross-validation results
    print("\n[INFO] Gerando visualizações dos resultados da validação cruzada...")
    plot_cv_results(results)

    # Treinar no conjunto completo para salvar o modelo final
    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "preprocessed_escolas.csv")
    X, y = load_data(data_path)
    print("[INFO] Dados carregados e alvo definido.")
    print(f"[INFO] Número de amostras: {X.shape[0]}, Número de características: {X.shape[1]}")

    # Split: 60% train, 30% test, 10% validation (holdout)
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42, stratify=y_temp
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Validation: {X_val.shape[0]}")

    # Dummy classifier on train set
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print("[INFO] Dummy Score (train):", dummy.score(X_train, y_train))
    print(y_train.count())
    print(y_train.value_counts())

    # Train model and cross-validation only on train set
    model = train_model_cv(X_train, y_train, cv_folds=3)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    print("[INFO] Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Final evaluation on holdout validation set (never seen before)
    y_val_pred = model.predict(X_val)
    print("[INFO] Holdout Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Salvar o modelo treinado
    model_path = os.path.join("models", "random_forest_model.joblib")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[INFO] Modelo salvo em {model_path}")
