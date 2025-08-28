# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:42:15 2025

@author: julio visualizacao 
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Carregando o Dataset ---
print("--- 1. Carregando o Dataset ---")
try:
    df_processado = pd.read_excel('meu_dataset_final_imputado_v2.xlsx')
    print("Dataset 'datase_saae.xlsx' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'datase_saae.xlsx' não foi encontrado.")
    exit()

# --- 2. Limpeza e Padronização dos Nomes das Colunas ---
print("\n--- 2. Limpando os nomes das colunas ---")
df_processado.columns = df_processado.columns.str.strip().str.upper()

# --- 3. Remoção de Dados Irrelevantes (BOOSTER) ---
print("\n--- 3. Removendo dados irrelevantes (BOOSTER) ---")
# Filtramos a linha que tem o nome 'BOOSTER' e criamos um novo DataFrame sem ela.
df_processado = df_processado[df_processado['RUA'] != 'BOOSTER'].copy()
print("Linhas com 'BOOSTER' removidas com sucesso.")


# --- 4. Conversão de Tipos de Dados ---
print("\n--- 4. Convertendo colunas para tipo numérico ---")
colunas_para_numericas = ['PH', 'COR', 'TURBIDEZ', 'CLORO', 'COLIFORMES_TOTAIS', 'E_COLI']
for col in colunas_para_numericas:
    if col in df_processado.columns:
        df_processado[col] = pd.to_numeric(df_processado[col], errors='coerce')


# --- 5. Engenharia de Features: Criando as Categorias de Ponto ---
print("\n--- 5. Criando categorias separadas para ETA e Mananciais ---")
lista_eta = ['ETA CERRADO']
lista_outros_mananciais = [
    'CLEMENTE ADUTORA', 'CLEMENTE REPRESA', 'CLEMENTE FUNDO',
    'ETA CERRADO SAÍDA', 'CANAL CLEMENTE', 'REPRESA IPANEMINHA', 'IPANEMINHA REPRESA'
]

condicao_eta = df_processado['RUA'].str.contains('|'.join(lista_eta), case=False, na=False)
condicao_outros = df_processado['RUA'].str.contains('|'.join(lista_outros_mananciais), case=False, na=False)

df_processado['TIPO_DE_PONTO_DETALHADO'] = np.where(
    condicao_eta, 'ETA_CERRADO',
    np.where(
        condicao_outros, 'Outros_Mananciais',
        'Residencial'
    )
)
print("Nova coluna 'TIPO_DE_PONTO_DETALHADO' criada com sucesso.")


# --- 6. Verificação e Visualização dos Dados Separados ---
# É aqui que a mágica acontece. Em vez de agrupar, vamos mostrar as linhas de verdade.
print("\n--- 6. Mostrando as linhas da sua análise ---")

# Mostra o número total de linhas e colunas
print("Tamanho do DataFrame final (linhas, colunas):", df_processado.shape)

# Filtra e mostra as linhas dos Mananciais para você ver
df_mananciais = df_processado[df_processado['TIPO_DE_PONTO_DETALHADO'] == 'Outros_Mananciais'].copy()
print("\n----- Linhas dos Outros Mananciais (Represa, etc.) -----")
print(df_mananciais[['RUA', 'TIPO_DE_PONTO_DETALHADO', 'PH', 'COR']].head(10)) # Mostra as 10 primeiras linhas

# Filtra e mostra as linhas do ETA CERRADO para você ver
df_eta = df_processado[df_processado['TIPO_DE_PONTO_DETALHADO'] == 'ETA_CERRADO'].copy()
print("\n----- Linhas do ETA CERRADO -----")
print(df_eta[['RUA', 'TIPO_DE_PONTO_DETALHADO', 'PH', 'COR']].head(10)) # Mostra as 10 primeiras linhas

# Filtra e mostra as linhas residenciais
df_residencial = df_processado[df_processado['TIPO_DE_PONTO_DETALHADO'] == 'Residencial'].copy()
print("\n----- Linhas Residenciais -----")
print(df_residencial[['RUA', 'TIPO_DE_PONTO_DETALHADO', 'PH', 'COR']].head(10)) # Mostra as 10 primeiras linhas

print("\n--- Script concluído com sucesso! ---")