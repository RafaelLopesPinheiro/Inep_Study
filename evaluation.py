import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    average_precision_score
)
from sklearn.model_selection import train_test_split

# Set style for better-looking plots
plt.style.use('ggplot')
sns.set_palette("husl")

def load_data(path):
    df = pd.read_csv(path)
    
    # Check if target already exists in preprocessed data
    if 'RISK_DROP_OUT' in df.columns:
        # Use existing target from preprocessing
        X = df.drop(columns=['RISK_DROP_OUT'])
        y = df['RISK_DROP_OUT']
    else:
        # Create target variable
        df['RISK_DROP_OUT'] = (df['QT_MAT_BAS_18_MAIS'] / (df['QT_MAT_BAS'] + 1)) > 0.15
        df['RISK_DROP_OUT'] = df['RISK_DROP_OUT'].astype(int)
        
        # Drop ALL enrollment-related columns that could leak information
        columns_to_drop = [
            'RISK_DROP_OUT', 
            'QT_MAT_BAS_18_MAIS', 
            'QT_MAT_BAS',
            # Add any other columns that might be derived from enrollment data
            # Check your preprocessed data for columns like:
            # 'QT_MAT_*', 'enrollment_ratio_*', etc.
        ]
        
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        y = df['RISK_DROP_OUT']
    
    return X, y

def plot_confusion_matrix(y_test, y_pred):
    """Enhanced confusion matrix with percentage and better styling"""
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=True, ax=ax1,
                square=True, linewidths=0.5)
    ax1.set_title("Matriz de Confusão - Contagens", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Predito", fontsize=12)
    ax1.set_ylabel("Real", fontsize=12)
    ax1.set_xticklabels(['Baixo Risco', 'Alto Risco'])
    ax1.set_yticklabels(['Baixo Risco', 'Alto Risco'])
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap="Oranges", cbar=True, ax=ax2,
                square=True, linewidths=0.5)
    ax2.set_title("Matriz de Confusão - Percentuais (%)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Predito", fontsize=12)
    ax2.set_ylabel("Real", fontsize=12)
    ax2.set_xticklabels(['Baixo Risco', 'Alto Risco'])
    ax2.set_yticklabels(['Baixo Risco', 'Alto Risco'])
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_scores):
    """Enhanced ROC curve with better styling and annotations"""
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=3, label=f"Modelo Random Forest (AUC = {roc_auc:.3f})", color='#2E86C1')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Classificador Aleatório")
    
    # Fill area under curve
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86C1')
    
    plt.xlabel("Taxa de Falsos Positivos", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    plt.title("Curva ROC - Predição de Risco de Evasão", fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add text box with interpretation
    textstr = f'AUC = {roc_auc:.3f}\n'
    if roc_auc > 0.9:
        textstr += 'Excelente'
    elif roc_auc > 0.8:
        textstr += 'Bom'
    elif roc_auc > 0.7:
        textstr += 'Razoável'
    else:
        textstr += 'Fraco'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.6, 0.2, textstr, fontsize=10, bbox=props)
    
    plt.tight_layout()
    plt.show()

def plot_precision_recall(y_test, y_scores):
    """Enhanced Precision-Recall curve with average precision"""
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=3, color='#E74C3C', 
             label=f'Modelo Random Forest (AP = {avg_precision:.3f})')
    
    # Fill area under curve
    plt.fill_between(recall, precision, alpha=0.2, color='#E74C3C')
    
    # Baseline (proportion of positive class)
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, 
                label=f'Baseline ({baseline:.3f})')
    
    plt.xlabel("Recall (Sensibilidade)", fontsize=12)
    plt.ylabel("Precision (Precisão)", fontsize=12)
    plt.title("Curva Precision-Recall - Predição de Risco de Evasão", fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance from the trained model"""
    # Extract feature importance
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importance = model.named_steps['classifier'].feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        
        # This is a simplified approach - you might need to adjust based on your preprocessing
        feature_names_processed = feature_names[:len(importance)]
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names_processed,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                       color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importância da Feature', fontsize=12)
        plt.title(f'Top {top_n} Features Mais Importantes', fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_class_distribution(y):
    """Plot the distribution of classes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    counts = y.value_counts()
    colors = ['#3498DB', '#E74C3C']
    bars = ax1.bar(['Baixo Risco', 'Alto Risco'], counts.values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_title('Distribuição das Classes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Escolas', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(counts.values, labels=['Baixo Risco', 'Alto Risco'], colors=colors, 
            autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
    ax2.set_title('Proporção das Classes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_summary(y_test, y_pred, y_scores):
    """Create a comprehensive metrics summary visualization"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': auc(*roc_curve(y_test, y_scores)[:2]),
        'AUC-PR': average_precision_score(y_test, y_scores)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Resumo das Métricas de Avaliação', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline = 0.5')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "preprocessed_escolas.csv")
    model_path = os.path.join("models", "random_forest_model.joblib")

    print("[INFO] Carregando dados e modelo...")
    X, y = load_data(data_path)
    model = joblib.load(model_path)

    print("[INFO] Visualizando distribuição das classes...")
    plot_class_distribution(y)

    # Divisão igual à do treinamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[INFO] Realizando predições...")
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    print("[INFO] Gerando visualizações...")
    plot_metrics_summary(y_test, y_pred, y_scores)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)
    plot_precision_recall(y_test, y_scores)
    
    # Plot feature importance (if available)
    try:
        plot_feature_importance(model, X.columns.tolist())
    except Exception as e:
        print(f"[WARNING] Não foi possível plotar importância das features: {e}")

    print("[INFO] Avaliação completa!")
