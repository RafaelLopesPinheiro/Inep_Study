import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_FILE_PATH = "data/processed/preprocessed_escolas.csv"
RAW_FILE_PATH = "data/raw/microdados_censo_escolar_2023/dados/microdados_ed_basica_2023.csv"

COLUMNS_TO_USE = [
    'CO_MUNICIPIO', 'NO_UF', 'TP_DEPENDENCIA', 'TP_LOCALIZACAO',
    'IN_BIBLIOTECA', 'IN_INTERNET', 'IN_SALA_PROFESSOR', 'IN_SALA_DIRETORIA', 'IN_QUADRA_ESPORTES',
    'QT_PROF_COORDENADOR', 'QT_PROF_PEDAGOGIA', 'QT_PROF_GESTAO', 'QT_PROF_PSICOLOGO', 'QT_PROF_ASSIST_SOCIAL',
    'IN_ACESSIBILIDADE_CORRIMAO', 'IN_BANHEIRO_PNE', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA',
    'QT_MAT_BAS', 'QT_MAT_MED', 'QT_MAT_FUND', 'QT_MAT_EJA', 'QT_MAT_BAS_15_17', 'QT_MAT_BAS_18_MAIS'
]

def preprocess_and_save_data(raw_file_path: str, output_file_path: str) -> pd.DataFrame:
    """Preprocessa os dados brutos e salva o resultado."""
    try:
        print("[INFO] Carregando dados brutos...")
        df = pd.read_csv(raw_file_path, sep=';', encoding='latin1', usecols=COLUMNS_TO_USE, low_memory=False)

        print("[INFO] Preenchendo valores ausentes...")
        df.fillna(0, inplace=True)

        print("[INFO] Convertendo categorias...")
        df['TP_DEPENDENCIA'] = df['TP_DEPENDENCIA'].astype('category')
        df['TP_LOCALIZACAO'] = df['TP_LOCALIZACAO'].astype('category')

        print("[INFO] Aplicando codificação one-hot...")
        df = pd.get_dummies(df, columns=['TP_DEPENDENCIA', 'TP_LOCALIZACAO'], drop_first=True)

        print("[INFO] Normalizando colunas quantitativas...")
        numeric_cols = [col for col in df.columns if col.startswith('QT_')]
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        print(f"[INFO] Salvando arquivo pré-processado em: {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False)

        return df

    except Exception as e:
        print(f"[ERRO] Falha ao processar os dados: {e}")
        return pd.DataFrame()

def load_or_preprocess_data() -> pd.DataFrame:
    """Carrega o dataset pré-processado ou realiza o pré-processamento se necessário."""
    if os.path.exists(PROCESSED_FILE_PATH):
        print("[INFO] Arquivo pré-processado encontrado. Carregando...")
        return pd.read_csv(PROCESSED_FILE_PATH)
    else:
        print("[INFO] Arquivo pré-processado não encontrado. Iniciando processamento...")
        return preprocess_and_save_data(RAW_FILE_PATH, PROCESSED_FILE_PATH)

if __name__ == "__main__":
    data = load_or_preprocess_data()
    print("[INFO] Pré-processamento finalizado.")
    print(data.info())
    print(data.head())
