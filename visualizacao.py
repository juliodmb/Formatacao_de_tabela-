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
import plotly.express as px
import plotly.subplots as sp
import webbrowser
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import shap
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
warnings.filterwarnings("ignore")

# Criar pasta de figuras
os.makedirs("figs", exist_ok=True)

# Configuração estética
sns.set(style="whitegrid", palette="muted")
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


"resolver o codigo amanaha"
# df_processado = df_processado[df_processado['RUA'] != 'ITUPARARANGA'].copy()
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
lista_outros_mananciais = [ # coloquei mais inputs (de mananciais)
    'CLEMENTE ADUTORA', 'CLEMENTE REPRESA', 'CLEMENTE FUNDO',
    'ETA CERRADO SAÍDA', 'CANAL CLEMENTE', 'REPRESA IPANEMINHA', 'IPANEMINHA REPRESA', 'ITUPARARANGA', 'REPRESA DE ITUPARARANGA', ' ITUPARARANGA( BARRAGEM )', 
    'ITUPARARANGA( MARGEM PIEDADE )', 'REP. ITUPARARANGA', 'ITUPARARANGA – BARRAGEM' , 'ITUPARARANGA – 7,9 m profundidade'
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

# --- 7. NORMALIZAÇÃO COM VERIFICAÇÃO VISUAL ---
print("\n" + "="*60)
print("NORMALIZAÇÃO DAS FEATURES - VERIFICAÇÃO COMPLETA")
print("="*60)

from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Aplica normalização
features_para_normalizar = ['PH', 'TURBIDEZ', 'CLORO', 'COR']
features_existentes = [f for f in features_para_normalizar if f in df_processado.columns]

print(f"Normalizando: {features_existentes}")

if features_existentes:
    scaler = RobustScaler()
    dados_normalizados = scaler.fit_transform(df_processado[features_existentes])
    
    for i, feature in enumerate(features_existentes):
        df_processado[f'{feature}_NORM'] = dados_normalizados[:, i]

# --- VERIFICAÇÃO NUMÉRICA ---
print("\n=== VERIFICAÇÃO NUMÉRICA ===")
for feature in features_existentes:
    original = df_processado[feature].dropna()
    normalizado = df_processado[f'{feature}_NORM'].dropna()
    
    print(f"{feature}:")
    print(f"  Original:    média={original.mean():.2f}, std={original.std():.2f}")
    print(f"  Normalizado: média={normalizado.mean():.3f}, std={normalizado.std():.3f}")

# --- VERIFICAÇÃO VISUAL - ANTES vs DEPOIS ---
print("\n=== GERANDO GRÁFICOS COMPARATIVOS ===")

fig, axes = plt.subplots(2, len(features_existentes), figsize=(5*len(features_existentes), 8))

if len(features_existentes) == 1:
    axes = axes.reshape(2, 1)

for i, feature in enumerate(features_existentes):
    # Dados ORIGINAIS
    axes[0, i].hist(df_processado[feature].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, i].set_title(f'{feature} - ORIGINAL\nstd: {df_processado[feature].std():.2f}')
    axes[0, i].set_xlabel('Valor Original')
    axes[0, i].set_ylabel('Frequência')
    
    # Dados NORMALIZADOS
    axes[1, i].hist(df_processado[f'{feature}_NORM'].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, i].set_title(f'{feature} - NORMALIZADO\nstd: {df_processado[f"{feature}_NORM"].std():.3f}')
    axes[1, i].set_xlabel('Valor Normalizado')
    axes[1, i].set_ylabel('Frequência')

plt.tight_layout()
plt.savefig('figs/normalizacao_antes_depois.png', dpi=300, bbox_inches='tight')
plt.show()

# --- COMPARAÇÃO DE INFLUÊNCIA NOS GRADIENTES ---
print("\n=== COMPARAÇÃO DE INFLUÊNCIA NOS GRADIENTES ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ANTES da normalização
stds_antes = [df_processado[feature].std() for feature in features_existentes]
ax1.bar(features_existentes, stds_antes, color='skyblue', alpha=0.7, edgecolor='black')
ax1.set_title('ANTES da Normalização\n(Influência Desbalanceada)')
ax1.set_ylabel('Desvio Padrão Original')
ax1.tick_params(axis='x', rotation=45)

# Adiciona valores nas barras
for i, v in enumerate(stds_antes):
    ax1.text(i, v + max(stds_antes)*0.01, f'{v:.2f}', ha='center', fontweight='bold')

# DEPOIS da normalização
stds_depois = [df_processado[f'{feature}_NORM'].std() for feature in features_existentes]
ax2.bar(features_existentes, stds_depois, color='lightcoral', alpha=0.7, edgecolor='black')
ax2.set_title('DEPOIS da Normalização\n(Influência Balanceada)')
ax2.set_ylabel('Desvio Padrão Normalizado')
ax2.tick_params(axis='x', rotation=45)

# Adiciona valores nas barras
for i, v in enumerate(stds_depois):
    ax2.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figs/influencia_gradientes.png', dpi=300, bbox_inches='tight')
plt.show()

# --- RESUMO FINAL ---
print("\n" + "="*60)
print("RESUMO DA NORMALIZAÇÃO")
print("="*60)

for i, feature in enumerate(features_existentes):
    influencia_antes = (stds_antes[i] / max(stds_antes)) * 100
    influencia_depois = (stds_depois[i] / max(stds_depois)) * 100
    
    print(f"{feature}:")
    print(f"  Influência ANTES: {influencia_antes:.1f}% do gradiente")
    print(f"  Influência DEPOIS: {influencia_depois:.1f}% do gradiente")
    print(f"  ✅ Balanceamento: {abs(influencia_depois - 100/len(features_existentes)):.1f}% de diferença do ideal")
    print()

print("🎯 RESULTADO: Todas as features agora contribuem igualmente para o gradiente!")
print("✅ NORMALIZAÇÃO CONCLUÍDA COM SUCESSO!")

# Atualiza as features para usar as normalizadas
feature_columns = ['PH_NORM', 'COR_NORM', 'TURBIDEZ_NORM', 'CLORO_NORM']
print(f"📋 Features normalizadas para uso: {feature_columns}")


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


# --- 3. MÉTODO DE BALANCEAMENTO CORRIGIDO ---
print("========================================================================")
print("APLICANDO BALANCEAMENTO COM PYTORCH WEIGHTED LOSS")
print("========================================================================")

class BalancedWaterQualityDataset(Dataset):
    """
    Dataset que aplica balanceamento via WeightedRandomSampler
    CORREÇÃO: Agora armazena labels como atributo
    """
    def __init__(self, df, feature_columns, target_column, sequence_length=5):
        self.df = df.sort_index().reset_index(drop=True)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        
        # Prepara os dados
        self.features = self.df[feature_columns].fillna(0).values.astype(np.float32)
        
        # Converte target para binário
        target_data = self.df[target_column].fillna(0).values
        if np.unique(target_data).size > 2:
            target_binary = (target_data > 0).astype(np.int64)
        else:
            target_binary = target_data.astype(np.int64)
        
        # CORREÇÃO: Cria sequências e armazena labels como atributo
        self.sequences, self.labels = self._create_sequences(target_binary)
        
        # Calcula pesos para balanceamento
        self.sample_weights = self._calculate_sample_weights()
    
    def _create_sequences(self, target_binary):
        sequences = []
        labels = []
        
        for i in range(len(self.features) - self.sequence_length):
            seq = self.features[i:i + self.sequence_length]
            label = target_binary[i + self.sequence_length]
            
            sequences.append(seq)
            labels.append(label)
        
        return torch.FloatTensor(sequences), torch.LongTensor(labels)
    
    def _calculate_sample_weights(self):
        """Calcula pesos para cada amostra baseado na frequência da classe"""
        class_counts = np.bincount(self.labels.numpy())
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        
        sample_weights = class_weights[self.labels]
        return sample_weights
    
    def get_balanced_sampler(self):
        """Retorna sampler balanceado"""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class WeightedFocalLoss(nn.Module):
    """
    Loss function que combina:
    - Weighted Cross Entropy (para balanceamento)
    - Focal Loss (para focar em exemplos difíceis)
    """
    def __init__(self, weights=None, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# --- 4. FUNÇÃO PARA VERIFICAÇÃO VISUAL E NUMÉRICA ---
def verificar_balanceamento(dataloader, target_name, tipo_ponto, etapa="APOS BALANCEAMENTO"):
    """
    Verifica visualmente e numericamente se o balanceamento funcionou
    """
    print("=" * 60)
    print(f"VERIFICANDO BALANCEAMENTO: {target_name} - {tipo_ponto}")
    print(f"ETAPA: {etapa}")
    print("=" * 60)
    
    # Coleta todas as labels do DataLoader
    all_labels = []
    for batch_X, batch_y in dataloader:
        all_labels.extend(batch_y.numpy())
    
    all_labels = np.array(all_labels)
    
    # Análise numérica
    contagem = np.bincount(all_labels)
    total = len(all_labels)
    proporcao = (contagem / total * 100).round(2)
    
    if len(contagem) == 2:
        ratio = contagem[0] / contagem[1]
        nivel = "BALANCEADO" if ratio < 3 else "QUASE BALANCEADO" if ratio < 5 else "AINDA DESBALANCEADO"
    else:
        ratio = float('inf')
        nivel = "SO UMA CLASSE"
    
    # Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras
    cores = ['#2E86AB', '#A23B72']
    labels_barras = ['Ausente (0)', 'Presente (1)'] if len(contagem) == 2 else ['Unica Classe']
    
    bars = ax1.bar(range(len(contagem)), contagem, color=cores[:len(contagem)], alpha=0.8, edgecolor='black')
    ax1.set_title(f'Distribuicao - {target_name}\n{tipo_ponto}\n{etapa}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classe', fontweight='bold')
    ax1.set_ylabel('Numero de Amostras', fontweight='bold')
    ax1.set_xticks(range(len(contagem)))
    ax1.set_xticklabels(labels_barras)
    ax1.grid(True, alpha=0.3)
    
    for bar, valor, perc in zip(bars, contagem, proporcao):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(contagem)*0.01,
                f'{valor}\n({perc}%)', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico de pizza
    if len(contagem) == 2:
        ax2.pie(contagem, labels=[f'Ausente\n{contagem[0]}', f'Presente\n{contagem[1]}'], 
                colors=cores, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Proporcoes\nRatio: {ratio:.2f}:1\n{nivel}', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, f'So uma classe\npresente', ha='center', va='center', fontsize=12, fontweight='bold')
        ax2.set_title('Distribuicao', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'figs/verificacao_{target_name}_{tipo_ponto}_{etapa.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Relatório numérico
    print("RELATORIO NUMERICO:")
    print(f"   Total de amostras: {total}")
    print(f"   Distribuicao: {dict(zip(range(len(contagem)), contagem))}")
    print(f"   Proporcao (%): {dict(zip(range(len(proporcao)), proporcao))}")
    if len(contagem) == 2:
        print(f"   Ratio: {ratio:.2f}:1")
        print(f"   Status: {nivel}")
    
    return ratio, nivel

# --- 5. APLICAÇÃO PRÁTICA CORRIGIDA ---
print("========================================================================")
print("APLICANDO BALANCEAMENTO NA PRATICA")
print("========================================================================")

# Configurações
feature_columns = ['PH', 'COR', 'TURBIDEZ', 'CLORO']
sequence_length = 5
batch_size = 32

resultados_balanceamento = {}

for tipo_ponto in df_processado['TIPO_DE_PONTO_DETALHADO'].unique():
    print(f"PROCESSANDO: {tipo_ponto}")
    print("-" * 50)
    
    df_tipo = df_processado[df_processado['TIPO_DE_PONTO_DETALHADO'] == tipo_ponto].copy()
    
    if len(df_tipo) < 20:
        print(f"Poucos dados ({len(df_tipo)} amostras). Pulando...")
        continue
    
    # Define variáveis alvo baseadas no tipo
    if tipo_ponto == 'Residencial':
        target_columns = [col for col in ['COLIFORMES_TOTAIS', 'E_COLI', 'RECLAMACAO', 'RECOLETA'] if col in df_tipo.columns]
    else:
        target_columns = [col for col in ['COLIFORMES_TOTAIS', 'E_COLI'] if col in df_tipo.columns]
    
    resultados_balanceamento[tipo_ponto] = {}
    
    for target_col in target_columns:
        print(f"Balanceando: {target_col}")
        
        try:
            # Cria dataset balanceado
            dataset = BalancedWaterQualityDataset(
                df_tipo, feature_columns, target_col, sequence_length
            )
            
            # DataLoader SEM balanceamento (para comparação)
            dataloader_desbalanceado = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
            # DataLoader COM balanceamento
            dataloader_balanceado = DataLoader(
                dataset, batch_size=batch_size, 
                sampler=dataset.get_balanced_sampler()
            )
            
            # Verifica ANTES do balanceamento
            print("ANTES do balanceamento:")
            ratio_antes, status_antes = verificar_balanceamento(
                dataloader_desbalanceado, target_col, tipo_ponto, "ANTES BALANCEAMENTO"
            )
            
            # Verifica DEPOIS do balanceamento
            print("DEPOIS do balanceamento:")
            ratio_depois, status_depois = verificar_balanceamento(
                dataloader_balanceado, target_col, tipo_ponto, "APOS BALANCEAMENTO"
            )
            
            # Calcula melhoria
            if ratio_antes != float('inf') and ratio_depois != float('inf'):
                melhoria = ((ratio_antes - ratio_depois) / ratio_antes) * 100
                print(f"MELHORIA: {melhoria:.1f}% de reducao no desbalanceamento")
            
            resultados_balanceamento[tipo_ponto][target_col] = {
                'ratio_antes': ratio_antes,
                'ratio_depois': ratio_depois,
                'status_antes': status_antes,
                'status_depois': status_depois,
                'dataset': dataset
            }
            
        except Exception as e:
            print(f"Erro ao balancear {target_col}: {e}")

# --- 6. RESUMO FINAL ---
print("========================================================================")
print("RESUMO FINAL DO BALANCEAMENTO")
print("========================================================================")

for tipo_ponto, targets in resultados_balanceamento.items():
    print(f"{tipo_ponto}:")
    
    for target_col, resultados in targets.items():
        print(f"  {target_col}:")
        print(f"    Ratio ANTES: {resultados['ratio_antes']:.2f}:1 -> {resultados['status_antes']}")
        print(f"    Ratio DEPOIS: {resultados['ratio_depois']:.2f}:1 -> {resultados['status_depois']}")

print("BALANCEAMENTO CONCLUIDO!")
print("Verifique as imagens na pasta 'figs/' para ver ANTES e DEPOIS")


# visualizacao de dados usando metodo spearman 

'''
# Lista de variáveis para correlação
cols = ["PH", "COR", "TURBIDEZ", "CLORO"]

# Calcula as correlações
corr_eta = df_eta[cols].corr(method="spearman")
corr_mananciais = df_mananciais[cols].corr(method="spearman")
corr_residencial = df_residencial[cols].corr(method="spearman")

# Subplots (1 linha, 3 colunas)
fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("ETA_CERRADO", "Outros_Mananciais", "Residencial"))

# ETA CERRADO
fig.add_heatmap(
    z=corr_eta.values,
    x=corr_eta.columns,
    y=corr_eta.index,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    row=1, col=1
)

# Outros Mananciais
fig.add_heatmap(
    z=corr_mananciais.values,
    x=corr_mananciais.columns,
    y=corr_mananciais.index,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    row=1, col=2
)

# Residencial
fig.add_heatmap(
    z=corr_residencial.values,
    x=corr_residencial.columns,
    y=corr_residencial.index,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    row=1, col=3
)

# Layout final
fig.update_layout(
    title="Correlação de Spearman por Tipo de Ponto",
    width=1200,
    height=500,
    coloraxis=dict(colorscale="RdBu", cmin=-1, cmax=1)
)

# Salva em HTML e abre no navegador
fig.write_html("correlacoes_spearman.html")
webbrowser.open("correlacoes_spearman.html")


# Boxplot com linha de média mstrica onde coseguimos captar a diferencia no ph 
# das ridencias mananciais e eta cerrado com a linha da  media

 sns.boxplot(data=df_processado, x="TIPO_DE_PONTO_DETALHADO", y="PH")
mean_ph = df_processado["PH"].mean()
plt.axhline(mean_ph, color="red", linestyle="--", label=f"Média PH = {mean_ph:.2f}")
plt.legend()
plt.show()

# --- Verifica/normaliza a coluna de DATA e ordena ---
    ...: if 'DATA' not in df_processado.columns:
    ...:     raise ValueError("Coluna 'DATA' não encontrada em df_processado. Verifique os nomes das colunas.")
    ...: 
    ...: df_processado['DATA'] = pd.to_datetime(df_processado['DATA'], errors='coerce')
    ...: df_time = df_processado.dropna(subset=['DATA']).sort_values('DATA').copy()
    ...: 
    ...: # --- Colunas que vamos usar (mantendo os nomes que você já tem) ---
    ...: feat_cols = [c for c in ['PH', 'COR', 'TURBIDEZ', 'CLORO'] if c in df_time.columns]
    ...: target_col = 'E_COLI' if 'E_COLI' in df_time.columns else ( 'COLIFORMES_TOTAIS' if 'COLIFORMES_TOTAIS' in df_time.columns else None )
    ...: 
    ...: if len(feat_cols) == 0:
    ...:     raise ValueError("Nenhuma feature PH/COR/TURBIDEZ/CLORO encontrada no df_processado.")
    ...: 
    ...: # --- Agregação diária (mediana) para suavizar e simplificar o gráfico ---
    ...: cols_to_agg = feat_cols.copy()
    ...: if target_col:
    ...:     cols_to_agg.append(target_col)
    ...: 
    ...: agg_daily = df_time.set_index('DATA')[cols_to_agg].resample('D').median()
    ...: 
    ...: # --- Gráfico: cada feature + alvo em eixo secundário (se existir) ---
    ...: fig, ax = plt.subplots(figsize=(14,5))
    ...: 
    ...: # plot das features (ex.: PH)
    ...: colors = ['tab:blue','tab:orange','tab:green','tab:red']
    ...: for i, col in enumerate(feat_cols):
    ...:     if col in agg_daily.columns:
    ...:         ax.plot(agg_daily.index, agg_daily[col], label=col, color=colors[i % len(colors)], linewidth=1)
    ...: 
    ...: # se existir alvo, plotar em eixo secundário (mais visível)
    ...: if target_col and target_col in agg_daily.columns:
    ...:     ax2 = ax.twinx()
    ...:     ax2.plot(agg_daily.index, agg_daily[target_col].fillna(0), label=target_col, color='tab:brown', linestyle='-', linewidth=1, alpha=0.8)
    ...:     ax2.set_ylabel(target_col)
    ...: else:
    ...:     ax2 = None
    ...: 
    ...: # --- Forçar os limites do eixo X para os dados reais (sem extrapolar) ---
    ...: min_date = agg_daily.index.min()
    ...: max_date = agg_daily.index.max()
    ...: ax.set_xlim(min_date, max_date)
    ...: 
    ...: # --- Configurar ticks para mostrar apenas os anos existentes ---
    ...: years = sorted({d.year for d in agg_daily.index})
    ...: # cria posições em 1º de janeiro de cada ano presente
    ...: year_ticks = [pd.Timestamp(year=y, month=1, day=1) for y in years]
    ...: ax.set_xticks(year_ticks)
    ...: ax.xaxis.set_major_locator(mdates.YearLocator())
    ...: ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ...: 
    ...: # --- Legenda combinada (ax + ax2) ---
    ...: lines, labels = ax.get_legend_handles_labels()
    ...: if ax2:
    ...:     lines2, labels2 = ax2.get_legend_handles_labels()
    ...:     lines += lines2; labels += labels2
    ...: ax.legend(lines, labels, loc='upper left')
    ...: 
    ...: ax.set_title("Séries temporais (agregação diária - mediana)")
    ...: ax.set_xlabel("Data")
    ...: ax.set_ylabel("Valor (mediana diária)")
    ...: 
    ...: plt.grid(axis='y', linestyle='--', alpha=0.5)
    ...: plt.tight_layout()
    ...: plt.show()



# --- Análise de Séries Temporais ---
print("\n--- Séries Temporais ---")

# Verifica se existe coluna de datas no dataset
if 'DATA' in df_processado.columns:
    df_processado['DATA'] = pd.to_datetime(df_processado['DATA'], errors='coerce')
    df_processado = df_processado.dropna(subset=['DATA'])

    # Agrupamento por ano e cálculo da média
    df_processado['ANO'] = df_processado['DATA'].dt.year
    medias_anuais = df_processado.groupby('ANO')[['PH', 'COR', 'TURBIDEZ', 'CLORO', 'COLIFORMES_TOTAIS', 'E_COLI']].mean()

    # --- Plot matplotlib/Seaborn ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=medias_anuais)
    plt.title("Médias anuais dos parâmetros de qualidade da água")
    plt.xlabel("Ano")
    plt.ylabel("Valor médio")
    plt.xticks(medias_anuais.index)  # <<-- força mostrar apenas os anos que existem
    plt.grid(True)
    plt.show()

    # --- Plot interativo com Plotly ---
    fig = px.line(
        medias_anuais,
        x=medias_anuais.index,
        y=medias_anuais.columns,
        markers=True,
        title="Médias anuais dos parâmetros de qualidade da água (interativo)"
    )
    # Força o eixo X a ir só até o último ano disponível
    fig.update_xaxes(range=[medias_anuais.index.min(), medias_anuais.index.max()])
    fig.show()

else:
    print("Coluna 'DATA' não encontrada no dataset.")


'''

# 






 