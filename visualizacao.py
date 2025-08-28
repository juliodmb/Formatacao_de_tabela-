# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:42:15 2025

@author: julio visualizacao 
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd
import numpy as np

# --- 1. Carregamento do Dataset ---
print("--- 1. Carregando o Dataset ---")
try:
    df_processado = pd.read_excel('datase_saae.xlsx')
    print("Dataset 'datase_saae.xlsx' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'datase_saae.xlsx' não foi encontrado.")
    exit()

# --- 2. Limpeza e Padronização dos Nomes das Colunas ---
print("\n--- 2. Limpando os nomes das colunas ---")
df_processado.columns = df_processado.columns.str.strip().str.upper()
print("Nomes das colunas limpos e padronizados para MAIÚSCULAS.")
print("Novas colunas:", df_processado.columns.tolist())


# --- 3. Conversão de Tipos de Dados ---
print("\n--- 3. Convertendo colunas para tipo numérico ---")
colunas_para_numericas = ['PH', 'COR', 'TURBIDEZ', 'CLORO', 'COLIFORMES_TOTAIS', 'E_COLI']
for col in colunas_para_numericas:
    if col in df_processado.columns:
        df_processado[col] = pd.to_numeric(df_processado[col], errors='coerce')
        print(f"Coluna '{col}' convertida para numérico.")


# --- 4. Engenharia de Features: Separando Pontos de Tratamento e Residências ---
print("\n--- 4. Criando nova feature para separar pontos de água e residências ---")
palavras_chave_saida = [
    'CLEMENTE ADUTORA', 'CLEMENTE REPRESA', 'CLEMENTE FUNDO', 'ETA CERRADO SAÍDA', 
     'CANAL CLEMENTE', 'REPRESA IPANEMINHA', 'IPANEMINHA REPRESA'
]
condicao_saida = df_processado['RUA'].str.contains('|'.join(palavras_chave_saida), case=False, na=False)
df_processado['TIPO_DE_PONTO'] = np.where(condicao_saida, 'Entrada/Saida_SAAE', 'Residencial')
print("Nova coluna 'TIPO_DE_PONTO' criada com sucesso.")


# --- 5. Criação de um DataFrame Filtrado para Análise Exploratória ---
print("\n--- 5. Criando um DataFrame exclusivo para endereços Residenciais ---")
df_residencial = df_processado[df_processado['TIPO_DE_PONTO'] == 'Residencial'].copy()
print("DataFrame 'df_residencial' criado com sucesso. Total de linhas:", len(df_residencial))
print("Primeiras 5 linhas do DataFrame Residencial:")
print(df_residencial.head())


# --- 6. Análise Exploratória: Agrupando por Bairro ---
print("\n--- 6. Realizando a análise (groupby) apenas nos dados residenciais ---")
agrupado_residencial = df_residencial.groupby('BAIRRO').agg({
    'PH': 'mean',
    'COR': 'mean',
    'COLIFORMES_TOTAIS': 'mean'
})
print(agrupado_residencial)


# --- 7. NOVA ANÁLISE: Separando apenas o ponto ETA CERRADO ---
print("\n--- 7. NOVA ANÁLISE: Agrupando por ETA CERRADO ---")

# Criamos uma nova lista com o nome específico
ETA = ['ETA CERRADO SAÍDA']
condicao_eta = df_processado['RUA'].str.contains('|'.join(ETA), case=False, na=False)
df_processado['TIPO_DE_PONTO_ETA'] = np.where(condicao_eta, 'ETA CERRADO', 'Outros_Pontos')

# Agora, fazemos o groupby usando a nova coluna
agrupado_eta = df_processado.groupby('TIPO_DE_PONTO_ETA').agg({
    'PH': 'mean',
    'COR': 'mean',
    'COLIFORMES_TOTAIS': 'mean'
})

print("\nAnálise detalhada do ponto ETA CERRADO:")
print(agrupado_eta)

print("\n--- Script concluído com sucesso! ---") 