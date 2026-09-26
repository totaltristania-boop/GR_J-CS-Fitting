import matplotlib

matplotlib.use('QtAgg')



import matplotlib.pyplot as plt

import warnings
import qualRpy.qualR as qr
import numpy as np
import unidecode
import pandas as pd
import unicodedata
import os

os.chdir(r'C:\Users\thela\maskNPF')

import glob

import shutil

from tqdm import tqdm

from PIL import Image

import matplotlib

from matplotlib import pyplot as plt

from matplotlib import ticker, colors, dates

from matplotlib.gridspec import GridSpec

from datetime import datetime, timedelta

import seaborn as sns

from itertools import combinations

from scipy.stats import gaussian_kde

from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import time as dtime



import torchvision

import torch

from torchvision import transforms



from utils import get_instance_segmentation_model

from utils import num2time, time2num, mkdirs

from utils import psd2im, draw_subplots, reshape_mask, mkdirs

from utils import get_SE, get_GR, get_GR_old, convert_matlab_time



import string

import sys

import subprocess



from os import listdir

from os.path import isfile, join

from scipy import stats



import torchvision



os.chdir(r'E:/totais')

path='E:/totais/txts_smps/10.2 - 429.4'



pd.options.display.max_columns = None



font = {'family': 'Arial'}

matplotlib.rc('font', **font)




%matplotlib qt

warnings.filterwarnings("ignore")

%load_ext autoreload
%autoreload 2


model = get_instance_segmentation_model()

modelfp = "C:/Users/thela/maskNPF/checkpoints/maskrcnnfull.pth"



# Verificar se CUDA está disponível

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Carregar o modelo para o dispositivo apropriado

model.load_state_dict(torch.load(modelfp, map_location=device))

model.to(device)

model.eval()

plt.rcParams.update({'font.size': 16})













folder_path = path

file_list = os.listdir(folder_path)

data_dict = {}




file_path_modas = 'E:/totais/Doutorado/fits_into_3modes4_fsp_full.txt'
index_path = 'E:/totais/txts_smps/saidas/OUTPUT/index_merged429.4.csv'

# Leia o arquivo de dados e o arquivo de índice

modas = pd.read_csv(file_path_modas, sep='\s+', encoding='unicode_escape', header=None)



df_index = pd.read_csv(index_path, encoding='unicode_escape')



# Converta a coluna de datas para o formato datetime

df_index['datetime'] = pd.to_datetime(df_index['datetime'], format='%d/%m/%Y %H:%M')

modas.index=df_index['datetime']

modas.rename(columns={0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'Concentração Moda de Nucleação', 

                      8: 'Concentração Moda de Aitken', 9: 'Concentração Moda de Acumulação', 

                      10: 'Diâmetro geométricos médio Nucleação', 11: 'Diâmetro geométricos médio Aitken', 

                      12: 'Diâmetro geométricos médio Acumulação', 13: 'Desvio-Padrão geométrico Nucleação', 

                      14: 'Desvio-Padrão geométrico Aitken', 15: 'Desvio-Padrão geométrico Acumulação', 

                      16: 'Rsquared', 17: 'dfe', 18: 'adjRsquare', 19: 'RMSE', 

                      21: 'Concentração total observada', 22: 'Concentração total ajustada'}, inplace=True)



columns_to_include = [
    'Diâmetro geométricos médio Nucleação',
    'Desvio-Padrão geométrico Nucleação',

    'Diâmetro geométricos médio Aitken',
    'Desvio-Padrão geométrico Aitken',

    'Diâmetro geométricos médio Acumulação',
    'Desvio-Padrão geométrico Acumulação'
]
       

# Subset the DataFrame with the selected columns

modas = modas[columns_to_include]

modas = modas.dropna(how='all')

modas.colunas=modas.columns.tolist()



dfs_por_dia_modas = []



# Iterar sobre os dias e gerar uma DataFrame para cada dia

for day, df_day in modas.groupby(modas.index.date):

    dfs_por_dia_modas.append(df_day)



dfs_por_dia_modas_todos = [
    df.copy()
    for df in dfs_por_dia_modas
]



file_path = 'E:/totais/txts_smps/sdistnew429_art_full.csv'
index_path = 'E:/totais/txts_smps/saidas/OUTPUT/index_merged429.4.csv'







# Leia o arquivo de dados e o arquivo de índice

df_ext = pd.read_csv(file_path, sep=',', encoding='unicode_escape',skiprows=1)

df_index = pd.read_csv(index_path, encoding='unicode_escape')









# 1) Remover o sufixo .1

# 1) Remover o sufixo .1

cols = df_ext.columns.str.replace(r'\.1$', '', regex=True)



# 2) Converter para float e arredondar

cols = np.round(cols.astype(float), 1)



# 3) Aplicar as colunas e remover duplicadas

df_ext.columns = cols

df_ext = df_ext.loc[:, ~df_ext.columns.duplicated()]



# Garantir que são strings depois

df_ext.columns = df_ext.columns.astype(str).str.strip()



# Converta a coluna de datas para o formato datetime

df_index['datetime'] = pd.to_datetime(df_index['datetime'], format='%d/%m/%Y %H:%M')

df_ext.index=df_index['datetime']

columns_to_drop=['10.2','0.0','2.6','94.7','98.2','105.5']

for column in columns_to_drop:

    try:

        df_ext.drop(column, axis=1, inplace=True)

    except KeyError:

        print(f"A coluna '{column}' não existe no DataFrame.")





df_ext = df_ext.reindex(sorted(df_ext.columns, key=float), axis=1)

df_ext = df_ext.apply(pd.to_numeric, errors='coerce')

df_ext.colunas=df_ext.columns.tolist()





df_ext.index = pd.to_datetime(df_ext.index)



dfs_por_dia = []



# Iterar sobre os dias e gerar uma DataFrame para cada dia

for day, df_day in df_ext.groupby(df_ext.index.date):

    dfs_por_dia.append(df_day)

# ==========================================================
# CÓPIA COM TODOS OS DIAS DA CAMPANHA
# ==========================================================

dfs_por_dia_todos = [
    df.copy()
    for df in dfs_por_dia
]

# Função para calcular Geometric Mean Diameter (GMD) diário

def get_GR_old(df, dp_min=9, dp_max=25, tm_res=10):



    # Get the date

    dt = str(df.index[0].date())

    

    # Get the values of the banana-shape region

    values = df.values

    

    # Get the dps

    dps = [float(dp) for dp in list(df.columns)]

    

    # Find the indexes of the start and end times

    col_min = np.where((np.array(dps) >= dp_min) == True)[0]

    col_max = np.where((np.array(dps) <= dp_max) == True)[0]

    

    # Check if valid range is found

    if len(col_min) == 0 or len(col_max) == 0:

        return None, None, None

    

    col_min = col_min[0]

    col_max = col_max[-1]

    

    # Obtain the time (index) when max concentration occurs for each dp

    peak_con = np.array([[get_max_time(values[:, i], df.index.values, tm_res), dps[i]]

                        for i in range(col_min, col_max+1)])

    

    # Drop the NaN values

    tm_con_mat = pd.DataFrame(peak_con).dropna(how='any').values

    

    # Split the time and dps

    x_tm = np.array([time2num(item) for item in tm_con_mat[:, 0]])

    y_dp = tm_con_mat[:, 1]

    

    # If there are less than 2 valid time-dp points, return None

    if len(x_tm) < 2:

        return None, None, None

    

    # Fit the time-dp relationship

    try:

        ransac = RANSACRegressor(random_state=42)

        ransac.fit(x_tm.reshape(-1, 1), y_dp)

        slope = ransac.estimator_.coef_[0]

        intercept = ransac.estimator_.intercept_

    except Exception:

        ols = LinearRegression()

        ols.fit(x_tm.reshape(-1, 1), y_dp)

        slope = ols.coef_[0]

        intercept = ols.intercept_

    

    # If the fitting slope less than 0, return None (negative GR)

    if slope <= 0:

        return None, None, None

    

    # Get the time and dp pairs for visualization

    num_min, num_max = (dp_min - intercept) / slope, (dp_max - intercept) / slope

    num_min, num_max = int(np.maximum(x_tm[0], num_min)), int(

        np.minimum(x_tm[-1], num_max))

    x_num = np.arange(num_min, num_max)

    x_pred = np.array([num2time(dt, item) for item in x_num])

    y_pred = np.array([slope*item+intercept for item in x_num])

    

    return slope*60, x_pred, y_pred







dfs_por_dia_copia = [df.copy() for df in dfs_por_dia]



# Suponha que dfs_por_dia_copia seja sua lista de DataFrames copiados

dfs_por_dia_resampled = []



for df in dfs_por_dia_copia:

    # Resample de soma por 60 minutos (60T)

    df_resampled = df.resample('60T').sum()

    

    # Adicionar o DataFrame resample à lista dfs_por_dia_resampled

    dfs_por_dia_resampled.append(df_resampled)



import numpy as np

import pandas as pd



# Constantes para o cálculo de C

lambda_air = 0.066e-6  # Caminho livre médio (em metros)



# Função para calcular o fator de correção C usando dp

def calcular_C(dp):

    return 1 + (((lambda_air) / dp) * (2.514 + (0.8 * np.exp((-0.55 * lambda_air) / dp))))



# Função para calcular k(dp)

def calcular_k_dp(dp):

    dp_m = dp * 1e-9    # Converter dp de nm para metros

    C = calcular_C(dp_m)

    return 3e-16 * C



# Função para calcular GR_coag para cada faixa de tamanho entre 9 e 600 nm

def calcular_gr_coag_por_faixa(df):

    gr_coag_values = []

    

    for index, row in df.iterrows():

        gr_coag_row = 0.0

        

        for col in df.columns:

            dp = float(col)

            

            if dp < 9 or dp > 600:

                continue  # Pular colunas fora do intervalo desejado

            

            k_dp = calcular_k_dp(dp)

            N = row[col]  # Concentração da faixa de tamanho dp

            

            if pd.isna(N):

                continue

            

            # Calcular GR_coag em nm/h

            gr_coag_value = (dp / 6) * k_dp * N   

            gr_coag_row += gr_coag_value

        

        gr_coag_values.append(gr_coag_row*1e9)

    

    return gr_coag_values



# Exemplo de uso

# Suponha que dfs_por_dia_resampled seja uma lista de DataFrames resampleados

# Vamos supor que dfs_por_dia_resampled já está definido



# Criar uma nova lista de DataFrames df_gr com o mesmo número de DataFrames que dfs_por_dia_resampled

df_gr = []



for df in dfs_por_dia:

    # Copiar o índice e as colunas do DataFrame original

    df_copy = df.copy()

    

    # Calcular GR_coag para cada linha do DataFrame original

    gr_coag_values = calcular_gr_coag_por_faixa(df)

    

    # Adicionar a coluna GR_coag ao DataFrame copiado

    df_copy['GR_coag'] = gr_coag_values

    

    # Adicionar o DataFrame modificado à lista df_gr

    df_gr.append(df_copy)



# Exemplo de acesso aos DataFrames resultantes em df_gr

for df in df_gr:

    print(df)

    

dfs_por_dia_resampled = []



for df in df_gr:

    # Resample de soma por 60 minutos (60T)

    df_resampled = df.resample('60T').mean()

    

    # Adicionar o DataFrame resample à lista dfs_por_dia_resampled

    dfs_por_dia_resampled.append(df_resampled)



# Carregar o CSV com as datas

csv_path = 'E:/totais/datas_npf2.csv'

df_dates = pd.read_csv(csv_path)



# Supondo que a coluna com as datas seja chamada 'date'

dates_to_filter = pd.to_datetime(df_dates['date']).dt.date



# Filtrar a lista de DataFrames com base nas datas extraídas

dfs_por_dia = [df for df in dfs_por_dia if pd.Series(df.index.date).isin(dates_to_filter).any()]

# Filtrar a lista de DataFrames com base nas datas extraídas

dfs_por_dia_modas = [df for df in dfs_por_dia_modas if pd.Series(df.index.date).isin(dates_to_filter).any()]















# Função para calcular o fator de correção C

def calcular_C(d):

    C = 1 + (lambda_air / d) * (2.514 + 0.8 * np.exp(-0.55 * d / lambda_air))

    return C



# Função para calcular k(dp)

def calcular_k_dp(d):

    C = calcular_C(d)

    return 3e-16 * C



# Função para calcular o GMD e a concentração total para o intervalo de 9 a 25 nm

def calcular_gmd_intervalo(
    df,
    dp_min=10.6,
    dp_max=25.0
):
    """
    Calcula:
      - GMD ponderado por número
      - concentração integrada

    para uma faixa de diâmetro.

    df deve estar em dN/dlogDp.
    """

    dps = pd.to_numeric(
        df.columns,
        errors='coerce'
    ).values.astype(float)

    mask = (
        np.isfinite(dps) &
        (dps >= dp_min) &
        (dps <= dp_max)
    )

    dps_sel = dps[mask]

    dados = df.iloc[
        :,
        np.where(mask)[0]
    ].copy()

    # ordenar por diâmetro
    ordem = np.argsort(dps_sel)

    dps_sel = dps_sel[ordem]
    dados = dados.iloc[:, ordem]

    logdp = np.log10(dps_sel)

    gmd = []
    N_integrado = []

    for _, row in dados.iterrows():

        y = pd.to_numeric(
            row,
            errors='coerce'
        ).values.astype(float)

        valid = np.isfinite(y)

        if valid.sum() < 2:
            gmd.append(np.nan)
            N_integrado.append(np.nan)
            continue

        x = logdp[valid]
        dp = dps_sel[valid]
        n = y[valid]

        # concentração integrada
        N = np.trapezoid(
            n,
            x
        )

        if (
            not np.isfinite(N)
            or N <= 0
        ):
            gmd.append(np.nan)
            N_integrado.append(np.nan)
            continue

        # GMD ponderado por número
        numerador = np.trapezoid(
            np.log(dp) * n,
            x
        )

        GMD = np.exp(
            numerador / N
        )

        gmd.append(GMD)
        N_integrado.append(N)

    return pd.DataFrame(
        {
            'GMD_10_25': gmd,
            'N_10_25': N_integrado
        },
        index=df.index
    )



# Função para calcular GR_coag

def calcular_gr_coag(df_gmd_10_25):

    gr_coag = []
    previous_time = None

    for index, row in df_gmd_10_25.iterrows():

        gmd_value = row['GMD_10_25']
        N_cm3 = row['N_10_25']

        

        if pd.isna(gmd_value) or pd.isna(N_cm3):

            gr_coag.append(np.nan)

            previous_time = index

            continue

        

        # Converter N de partículas/cm³ para partículas/m³

        N_m3 = N_cm3 * 1e6

        

        gmd_value_meters = gmd_value * 1e-9  # Converter nm para metros

        k_dp = calcular_k_dp(gmd_value_meters)

        

        # Calcular o intervalo de tempo (Δt) em horas

        if previous_time is not None:

            delta_t = (index - previous_time).total_seconds() / 3600  # Convertendo segundos para horas

        else:

            delta_t = 0

        

        # Ajustar GR_coag considerando Δt

        gr_coag_value = (gmd_value / 6) * k_dp * N_m3 * delta_t 

        gr_coag_value = gr_coag_value * 1e9 / 3600

        gr_coag.append(gr_coag_value)

        

        previous_time = index

    

    gr_coag_df = pd.DataFrame({'GR_coag': gr_coag}, index=df_gmd_10_25.index)

    gr_coag_df = gr_coag_df.dropna()  # Remove valores NaN

    return gr_coag_df



# Iterar sobre cada DataFrame e calcular GR_coag

gr_coag_list = []



for df in dfs_por_dia:

    df_gmd_10_25 = calcular_gmd_intervalo(
        df,
        dp_min=10.6,
        dp_max=25.0
    )

    gr_coag_df = calcular_gr_coag(
        df_gmd_10_25
    )

    gr_coag_list.append(
        gr_coag_df
    )



# Calcular a média de GR_coag para cada DataFrame e imprimir

for i, gr_coag_df in enumerate(gr_coag_list):

    gr_coag_mean = gr_coag_df['GR_coag'].mean()

    print(f"Dia {i+1}: Média de GR_coag = {gr_coag_mean:.8f} nm/h")





# Constantes físicas

k = 1.38e-23  # Constante de Boltzmann, J/K

T = 298  # Temperatura, K

eta = 1.81e-5  # Viscosidade do ar, Pa.s

lambda_air = 0.066e-6   # Caminho livre médio do ar, m



# Função para calcular o fator de correção de Cunningham

def calcular_C(d):

    C = 1 + (lambda_air / d) * (2.514 + 0.8 * np.exp(-0.55 * d / lambda_air))

    return C



# Função para calcular o coeficiente de difusão

def diffusion_coefficient(d_p):

    C = calcular_C(d_p)

    return (k * T * C) / (3 * np.pi * eta * d_p)



# Função para calcular k(d_p,j)

def coagulation_kernel(d_p_i, d_p_j):

    D_i = diffusion_coefficient(d_p_i)

    D_j = diffusion_coefficient(d_p_j)

    beta = 1.0  # Aproximação para o fator de correção

    return (D_i + D_j) * np.pi * beta



# Função para calcular CoagS_i para um DataFrame

def calculate_coags(df):

    particle_diameter = pd.to_numeric(df.columns, errors='coerce')   # Converter para metros

    number_concentration = df.values



    coagS = []

    for i in range(len(particle_diameter)):

        d_p_i = particle_diameter[i]

        N_i = number_concentration[:, i]



        sum_coag = 0

        for j in range(len(particle_diameter)):

            d_p_j = particle_diameter[j]

            N_j = number_concentration[:, j]



            k_dpij = coagulation_kernel(d_p_i, d_p_j)

            sum_coag += k_dpij * N_j



        coagS.append(sum_coag * N_i.mean())  # Média da concentração numérica



    return np.array(coagS)



# Função para calcular GR_scav

def calculate_gr_scav(df, coagS):

    particle_diameter = pd.to_numeric(df.columns, errors='coerce').to_numpy()   # Converter para metros

    number_concentration = df.values



    N = number_concentration.mean(axis=0)

    d_p_star = particle_diameter.mean()



    coagS = np.array(coagS)



    sum1 = np.sum(coagS * N)

    sum2 = np.sum(coagS * particle_diameter * N)



    gr_scav = d_p_star * (sum1 / sum2)  # Corrigido cálculo



    # Conversão de unidades: m/h para nm/h

    gr_scav_nmh = gr_scav 



    return gr_scav_nmh 





# Calcular CoagS para cada DataFrame

coagS_por_dia = [calculate_coags(df) for df in dfs_por_dia]



# Calculate GR_scav for each DataFrame

gr_scav_por_dia = [calculate_gr_scav(df, coagS.sum(axis=1)) for df, coagS in zip(dfs_por_dia, coagS_por_dia)]



# Imprimir os resultados

for i, gr_scav in enumerate(gr_scav_por_dia):

    print(f"Dia {i+1}: GR_scav (nm/h) = {gr_scav:.8f}")

    























##################################################################################







from sklearn.linear_model import LinearRegression, RANSACRegressor

model = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=2,
    residual_threshold=5.0,
    random_state=0
)



import matplotlib.dates as mdates

from matplotlib.widgets import Button, SpanSelector

from sklearn.linear_model import LinearRegression

import pandas as pd

import numpy as np

import matplotlib.colors as colors

from matplotlib.offsetbox import OffsetImage, AnnotationBbox



selected_data_list = []

current_class = None

stop_flag = False

text_boxes = []

delta_t_box = None






def calcular_pico_intervalo(
    df,
    dp_min=10.6,
    dp_max=25.0,
    nome='Dp_peak_10_25'
):
    """
    Para cada scan, encontra o bin de maior concentração
    dentro da faixa dp_min–dp_max.

    Uso diagnóstico para visualizar a população sub-25 nm.
    NÃO é o Maximum-Concentration Method usado para GR.
    """

    dps = pd.to_numeric(
        df.columns,
        errors='coerce'
    ).to_numpy(dtype=float)

    mask = (
        np.isfinite(dps) &
        (dps >= dp_min) &
        (dps <= dp_max)
    )

    dps_sel = dps[mask]

    dados = (
        df.iloc[:, np.where(mask)[0]]
        .apply(pd.to_numeric, errors='coerce')
    )

    resultado = []

    for _, row in dados.iterrows():

        valores = row.to_numpy(dtype=float)

        if np.isfinite(valores).any():
            idx = np.nanargmax(valores)
            resultado.append(dps_sel[idx])
        else:
            resultado.append(np.nan)

    return pd.DataFrame(
        {nome: resultado},
        index=df.index
    )




def calcular_max_conc(df):

    # Calcula o índice do diâmetro correspondente à concentração máxima para cada linha

    max_indices = np.argmax(df.values, axis=1)

    

    # Usa esses índices para obter o diâmetro correspondente ao valor máximo

    max_concentration_diameters = df.columns[max_indices].astype(float)

    

    # Cria o DataFrame com o valor do diâmetro máximo para cada timestamp

    max_concentration_df = pd.DataFrame({'GMD': max_concentration_diameters}, index=df.index)

    

    return max_concentration_df





class CustomButton:

    def __init__(self, ax, label, color='0.85', hovercolor='0.95'):

        self.ax = ax

        self.label = label

        self.color = color

        self.hovercolor = hovercolor

        self.button = Button(ax, label, color=color, hovercolor=hovercolor)

        self.default_color = color



    def on_clicked(self, func):

        self.button.on_clicked(func)



    def reset_color(self):

        self.button.color = self.default_color

        self.ax.set_facecolor(self.default_color)



    def set_selected(self):

        self.button.color = 'lightgreen'

        self.ax.set_facecolor('lightgreen')



def plot_regression(ax, data, color='red', robust=False):
    if data.empty or data.shape[0] < 2:
        return None

    X = ((data.index - data.index[0]).total_seconds().values / 3600).reshape(-1, 1)
    y = data.iloc[:, 0].values

    if robust:
        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=2,
            residual_threshold=5.0,
            random_state=0
        )
        model.fit(X, y)
        slope = model.estimator_.coef_[0]
        y_pred = model.predict(X)
    else:
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        y_pred = model.predict(X)

    ax.plot(data.index, y_pred, '--', color=color, linewidth=2)

    return slope

def calcular_gr_max_concentracao(
    df_evento,
    dp_min=10.6,
    dp_max=25.0,
    smooth_minutes=10
):
    """
    Maximum-Concentration Method.

    Para cada bin de diâmetro entre dp_min e dp_max,
    encontra o horário em que a concentração atinge
    seu máximo dentro do intervalo selecionado.

    Depois ajusta Dp = a + GR * t.
    """

    dps = pd.to_numeric(
        df_evento.columns,
        errors='coerce'
    ).to_numpy(dtype=float)

    mask = (
        np.isfinite(dps) &
        (dps >= dp_min) &
        (dps <= dp_max)
    )

    indices = np.where(mask)[0]
    dps_sel = dps[mask]

    pontos = []

    for idx_col, dp in zip(
        indices,
        dps_sel
    ):

        serie = pd.to_numeric(
            df_evento.iloc[:, idx_col],
            errors='coerce'
        ).dropna()

        if len(serie) < 3:
            continue

        # Suavização temporal para evitar que
        # um único scan determine t_max
        if smooth_minutes is not None:

            serie_proc = serie.rolling(
                f'{smooth_minutes}min',
                center=True,
                min_periods=2
            ).mean()

        else:
            serie_proc = serie

        serie_proc = serie_proc.dropna()

        if serie_proc.empty:
            continue

        t_max = serie_proc.idxmax()

        pontos.append(
            {
                't_max': t_max,
                'Dp': dp,
                'conc_max': serie_proc.loc[t_max]
            }
        )

    pontos = pd.DataFrame(pontos)

    if len(pontos) < 3:
        return None

    pontos = pontos.sort_values(
        't_max'
    ).reset_index(drop=True)

    t_ref = pontos['t_max'].min()

    x = (
        (
            pontos['t_max'] - t_ref
        )
        .dt.total_seconds()
        .to_numpy()
        / 3600
    ).reshape(-1, 1)

    y = pontos['Dp'].to_numpy()

    model = LinearRegression()

    model.fit(
        x,
        y
    )

    GR = model.coef_[0]

    R2 = model.score(
        x,
        y
    )

    # reta apenas dentro do intervalo dos pontos MC
    t_linha = pd.date_range(
        pontos['t_max'].min(),
        pontos['t_max'].max(),
        periods=100
    )

    x_linha = (
        (
            t_linha - t_ref
        ).total_seconds()
        / 3600
    ).to_numpy().reshape(-1, 1)

    dp_linha = model.predict(
        x_linha
    )

    return {
        'GR': GR,
        'R2': R2,
        'pontos': pontos,
        't_linha': t_linha,
        'dp_linha': dp_linha
    }

# ==============================================================================
# CALCULAR GR COM RASTREIO INTELIGENTE ("CORTE NO TETO")
# ==============================================================================
def calcular_gr_dpg(serie_dpg, start_dt, end_dt, dp_min, dp_max, truncar_teto=True):
    serie = pd.to_numeric(serie_dpg, errors='coerce').copy()
    serie.index = pd.to_datetime(serie.index)
    if serie.index.tz is not None:
        serie.index = serie.index.tz_localize(None)

    # Recorte inicial da janela desenhada pelo usuário
    serie = serie.loc[(serie.index >= start_dt) & (serie.index <= end_dt)]
    
    if truncar_teto:
        pontos_validos = []
        rastreando = False
        
        for t, val in serie.items():
            if pd.isna(val):
                continue
            
            # Se a partícula está dentro da moda, começamos a guardar os pontos
            if dp_min <= val <= dp_max:
                rastreando = True
                pontos_validos.append(t)
            # Se furou o teto (ex: passou de 25nm) e já estávamos a rastrear, PARA TUDO!
            elif val > dp_max and rastreando:
                break
                
        serie = serie.loc[pontos_validos]
    else:
        # Para o Global, apenas ignora o corte no teto
        serie = serie[(serie >= dp_min) & (serie <= dp_max)].dropna()

    if len(serie) < 3:
        return None

    t_ref = serie.index.min()
    x = ((serie.index - t_ref).total_seconds() / 3600).to_numpy().reshape(-1, 1)
    y = serie.to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    t_linha = pd.date_range(serie.index.min(), serie.index.max(), periods=100)
    x_linha = ((t_linha - t_ref).total_seconds() / 3600).to_numpy().reshape(-1, 1)

    return {
        'GR': model.coef_[0],
        'R2': model.score(x, y),
        'pontos': serie,
        't_linha': t_linha,
        'dp_linha': model.predict(x_linha)
    }




def save_selected_data():

    df_selected_data = pd.DataFrame(selected_data_list, columns=['start_datetime', 'end_datetime', 'GRs', 'classe'])

    df_selected_data.to_csv('selected_data.csv', index=False)

    print("Dados selecionados salvos em 'selected_data.csv'.")



def stop_annotation(event):

    global stop_flag

    stop_flag = True

    save_selected_data()

    plt.close(fig)

    print("Anotação finalizada.")



def classify_data(classe, button):
    global current_class, class_buttons
    print(selected_data_list)

    if selected_data_list:

        selected_data_list[-1] = (*selected_data_list[-1][:3], classe)

        current_class = classe

        for btn in class_buttons:

            if btn != button:

                btn.reset_color()

        button.set_selected()

    else:

        print("Nenhum dado para classificar.")



def next_without_saving(event):

    global current_class, fig

    if selected_data_list:

        selected_data_list[-1] = (*selected_data_list[-1][:3], "Sem eventos/Não classificado")

    plt.close(fig)



def next_with_saving(event):

    if selected_data_list and selected_data_list[-1][3] is None:

        print("Por favor, selecione uma classe antes de continuar.")

    else:

        plt.close(fig)







def toggle_image_visibility(event):

    global ab

    ab.set_visible(not ab.get_visible())

    fig.canvas.draw_idle()





        

      
def calcular_dp_peak_global(df):

    valores = df.to_numpy(dtype=float)

    dps = pd.to_numeric(
        df.columns,
        errors='coerce'
    ).to_numpy(dtype=float)

    resultado = np.full(
        len(df),
        np.nan
    )

    for j, row in enumerate(valores):

        valid = np.isfinite(row)

        if valid.any():

            idx_validos = np.where(valid)[0]

            idx_local = np.nanargmax(
                row[valid]
            )

            idx_global = idx_validos[idx_local]

            resultado[j] = dps[idx_global]

    return pd.Series(
        resultado,
        index=df.index,
        name='Dp_peak'
    )  





# ==============================================================================
# SALVAMENTO INTELIGENTE BASEADO NA CLASSE
# ==============================================================================
def save_evento_completo(event):
    global selected_data_list
    if not selected_data_list:
        return
        
    evento = selected_data_list[-1]
    start_dt, end_dt, gr_info, classe = evento[:4]
    
    if classe is None:
        print("ERRO: Escolha uma classe (I, Ib, II ou Indeterminado) primeiro!")
        return
        
    if not isinstance(gr_info, dict):
        print("Evento já salvo.")
        return
        
    # Lógica inteligente: Guarda só o que interessa para cada classe
    if classe in ['Classe I', 'Classe Ib', 'Classe II']:
        dados_salvos = {
            'GR_Global': gr_info.get('GR_Global'),
            'GR_10_25': gr_info.get('GR_10_25'),
            'Start_Nuc': gr_info.get('Start_Nuc'),
            'End_Nuc': gr_info.get('End_Nuc')
        }
    elif classe == 'Indeterminado':
        dados_salvos = {
            'GR_Global': gr_info.get('GR_Global'),
            'GR_25_100': gr_info.get('GR_25_100'),
            'Start_Aitken': gr_info.get('Start_Aitken'),
            'End_Aitken': gr_info.get('End_Aitken')
        }
    else:
        dados_salvos = gr_info
        
    selected_data_list[-1] = (start_dt, end_dt, str(dados_salvos), classe)
    save_selected_data()
    print(f"-> SUCESSO! Evento {classe} salvo de forma otimizada!")






current_class = None

stop_flag = False

text_boxes = []

delta_t_box = None




user_name ='rubens.pereira@usp.br' 
my_password='sakuracc' 
stations_codes = [] 
poluttant_codes = [] 
station_list=pd.DataFrame(qr.cetesb_aqs()) 
param_list=pd.DataFrame(qr.cetesb_param()) 
# ========================================================
# FUNÇÃO ORIGINAL RESTAURADA
# ========================================================
def coleta_e_printa(path):
    infos = pd.read_csv(path, sep=';', encoding='unicode-escape',
                        quotechar='"', decimal=".") 
    stations_codes=infos["Estações Codes"]
    poluttant_codes=infos["Param Codes"]
    start_date=infos["Ini_Date"][0]
    end_date=infos["Final_Date"][0]
    stations_codes=stations_codes.dropna()
    poluttant_codes=poluttant_codes.dropna()
    
                    
    tamanho_dataframe=int(len(stations_codes)*len(poluttant_codes))
    all_data = [[] for i in range(tamanho_dataframe)]
    tamanho_dataframe=0
    for i in range (len(stations_codes)):
        for k in tqdm(range (len(poluttant_codes))):
       
    
            o3_code = poluttant_codes[k] 
            pin_code = int(stations_codes[i])
            station_index=int((station_list[station_list['code']==stations_codes[i]].index.values.astype(int)[0]))
            param_index=int((param_list[param_list['code']==poluttant_codes[k]].index.values.astype(int)[0]))
            format_ini_date=pd.to_datetime(start_date , format="%d/%m/%Y").strftime('%Y-%m-%d')
            format_final_date=pd.to_datetime(end_date, format="%d/%m/%Y").strftime('%Y-%m-%d')
            print('\n')
            print("Adquirindo dados da estação " , station_list['name'][station_index])
            print("Para o composto/parametro ",param_list['name'][param_index])
            nome_param=str(param_list['name'][param_index])
            nome_esta=str(station_list['name'][station_index])
            
          
            dados_estacao = qr.cetesb_retrieve(
              user_name,
              my_password,
              start_date,
              end_date,
              o3_code,
              pin_code
              )
            
            df= dados_estacao
            if(df.isnull().values.all()!= True):
                df = df.drop(["name","pol_name","units"] ,axis=1)
                df=df.dropna(axis=0)
                df = df.rename(columns={'day': 'Dia','hour': 'Hora','val': nome_param})
                df["DateTime"] = df.index
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df['Mes-Ano'] = df['DateTime'].dt.strftime('%Y-%m')
                df['Ano'] = df['DateTime'].dt.strftime('%Y')
                df['mes'] = df['DateTime'].dt.strftime('%b')
                df['estação']=str(station_list['name'][station_index])
                all_data[tamanho_dataframe].append(df)
                tamanho_dataframe = (tamanho_dataframe+1)
            
    return all_data

if 'dados_polu' not in locals() or 'dados_polu' not in globals():
    dados_polu = coleta_e_printa("E:/totais/dados_polu.csv")
def normalizar_texto(txt):
    txt = str(txt)
    txt = ''.join(
        c for c in unicodedata.normalize('NFKD', txt)
        if not unicodedata.combining(c)
    )
    return txt.lower().strip()


def encontrar_o3(dados_polu):

    encontrados = []

    colunas_auxiliares = {
        'dia',
        'hora',
        'datetime',
        'mes-ano',
        'ano',
        'mes',
        'estacao',
        'estação',
        'units',
        'name',
        'pol_name'
    }

    # dados_polu é uma lista de listas
    for grupo in dados_polu:

        if not grupo:
            continue

        for df_temp in grupo:

            if df_temp is None or df_temp.empty:
                continue

            df_temp = df_temp.copy()

            # garante datetime
            if 'DateTime' in df_temp.columns:
                idx = pd.to_datetime(
                    df_temp['DateTime'],
                    errors='coerce'
                )
            else:
                idx = pd.to_datetime(
                    df_temp.index,
                    errors='coerce'
                )

            df_temp.index = idx

            for col in df_temp.columns:

                nome = normalizar_texto(col)

                if nome in colunas_auxiliares:
                    continue

                # aceita O3, Ozônio, Ozonio, Ozone...
                if (
                    nome == 'o3'
                    or 'ozon' in nome
                ):

                    serie = pd.to_numeric(
                        df_temp[col],
                        errors='coerce'
                    )

                    serie.index = df_temp.index
                    serie = serie.sort_index()

                    if 'estação' in df_temp.columns:
                        estacao = df_temp['estação'].dropna()

                        if len(estacao) > 0:
                            estacao = estacao.iloc[0]
                        else:
                            estacao = 'desconhecida'
                    else:
                        estacao = 'desconhecida'

                    encontrados.append({
                        'estacao': estacao,
                        'parametro': col,
                        'serie': serie
                    })

    return encontrados


o3_disponiveis = encontrar_o3(dados_polu)

print("\nDados de O3 encontrados:")

for j, item in enumerate(o3_disponiveis):
    print(
        j,
        " | estação:",
        item['estacao'],
        " | parâmetro:",
        item['parametro']
    )


# ==========================================================
# DADOS DE IRRADIÂNCIA
# ==========================================================

caminho_meteo = "E:/totais/windy.csv"

meteo_diag = pd.read_csv(
    caminho_meteo,
    sep=';',
    encoding='unicode_escape',
    on_bad_lines='skip',
    index_col='Datetime'
)

meteo_diag.index = pd.to_datetime(
    meteo_diag.index,
    format='%d/%m/%Y %H:%M'
)

# Corrigir somente xx:59
mask59 = meteo_diag.index.minute == 59

meteo_diag.index = meteo_diag.index.where(
    ~mask59,
    meteo_diag.index + pd.DateOffset(minutes=1)
)

meteo_diag = meteo_diag.replace(-9999, np.nan)

irradiancia = pd.to_numeric(
    meteo_diag['Irradiancia solar'],
    errors='coerce'
)

# Pelo seu arquivo, você vinha usando a irradiância invertida
irradiancia = irradiancia * -1

irradiancia = irradiancia.sort_index()




def integrar_pnc(df, dp_min, dp_max):
    """
    Integra dN/dlogDp entre dp_min e dp_max.

    Retorna Series em #/cm³.
    """

    dps = pd.to_numeric(
        df.columns,
        errors='coerce'
    ).values

    mask = (
        np.isfinite(dps) &
        (dps >= dp_min) &
        (dps <= dp_max)
    )

    dps_sel = dps[mask]

    if len(dps_sel) < 2:
        return pd.Series(
            np.nan,
            index=df.index
        )

    dados = df.iloc[:, np.where(mask)[0]].copy()

    # Garante ordem crescente
    ordem = np.argsort(dps_sel)

    dps_sel = dps_sel[ordem]
    dados = dados.iloc[:, ordem]

    dlogDp = np.diff(
        np.log10(dps_sel)
    )

    valores = dados.values

    integral = np.nansum(
        valores[:, :-1] * dlogDp,
        axis=1
    )

    # se uma linha inteira estiver faltando,
    # não transformar em zero
    linhas_sem_dados = np.all(
        ~np.isfinite(valores[:, :-1]),
        axis=1
    )

    integral[linhas_sem_dados] = np.nan

    return pd.Series(
        integral,
        index=df.index
    )




# ==========================================================
# LOCALIZAR E PREPARAR SÉRIE DE O3
# ==========================================================

o3_disponiveis = encontrar_o3(dados_polu)

print("\nDados de O3 encontrados:")

for j, item in enumerate(o3_disponiveis):
    print(
        j,
        "| estação:",
        item['estacao'],
        "| parâmetro:",
        item['parametro'],
        "| início:",
        item['serie'].index.min(),
        "| fim:",
        item['serie'].index.max()
    )

if len(o3_disponiveis) == 0:
    raise ValueError("Nenhuma série de O3 foi encontrada em dados_polu.")

# Escolha qual série usar
INDICE_O3 = 0

serie_o3 = o3_disponiveis[INDICE_O3]['serie'].copy()

serie_o3.index = pd.to_datetime(serie_o3.index)
serie_o3 = pd.to_numeric(serie_o3, errors='coerce')

serie_o3 = serie_o3[
    ~serie_o3.index.duplicated(keep='first')
]

serie_o3 = serie_o3.sort_index()

print("\nSérie de O3 escolhida:")
print("Estação:", o3_disponiveis[INDICE_O3]['estacao'])
print("Parâmetro:", o3_disponiveis[INDICE_O3]['parametro'])
print(serie_o3.head())

from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.cm import ScalarMappable


# ==========================================================
# PREPARAÇÃO DO VENTO
# ==========================================================

def circular_mean_deg(series):
    """
    Média circular de direção meteorológica.
    """
    s = pd.to_numeric(series, errors='coerce').dropna()

    if len(s) == 0:
        return np.nan

    rad = np.deg2rad(s.values)

    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))

    ang = np.rad2deg(
        np.arctan2(sin_mean, cos_mean)
    )

    if ang < 0:
        ang += 360

    return ang


# Se meteo_diag já foi lido para irradiância,
# podemos reutilizar o mesmo DataFrame
meteo_wind = meteo_diag.copy()

meteo_wind['ws'] = pd.to_numeric(
    meteo_wind['ws'],
    errors='coerce'
)

meteo_wind['wd'] = pd.to_numeric(
    meteo_wind['wd'],
    errors='coerce'
)

# Média horária:
# velocidade = média normal
# direção = média circular
wind_hour = pd.DataFrame({

    'ws':
        meteo_wind['ws']
        .resample('1h')
        .mean(),

    'wd_deg':
        meteo_wind['wd']
        .resample('1h')
        .apply(circular_mean_deg)
})

wind_hour = wind_hour.sort_index()


# ==========================================================
# ESCALA DE COR GLOBAL
# ==========================================================
# Importante: mesma cor = mesma velocidade em TODOS os dias
# 99º percentil evita que um único valor extremo destrua a escala.

WIND_VMAX = np.nanpercentile(
    wind_hour['ws'].dropna(),
    99
)

if not np.isfinite(WIND_VMAX) or WIND_VMAX <= 0:
    WIND_VMAX = 5.0

wind_norm = colors.Normalize(
    vmin=0,
    vmax=WIND_VMAX
)

wind_cmap = plt.get_cmap('turbo')



# ==========================================================
# MARCADOR DE SETA FIXO
# ==========================================================

# A seta-base aponta para a direita (Leste)
ARROW_MARKER = Path([
    (-0.50, -0.11),
    ( 0.10, -0.11),
    ( 0.10, -0.29),
    ( 0.52,  0.00),
    ( 0.10,  0.29),
    ( 0.10,  0.11),
    (-0.50,  0.11),
    (-0.50, -0.11)
])


def plot_wind_panel(
    ax,
    wind_day,
    norm=wind_norm,
    cmap=wind_cmap,
    calm_threshold=0.3,
    passo=2,
    marker_size=230
):
    """
    Painel compacto de vento.

    - direção da seta = direção para onde o ar se desloca
    - cor = velocidade
    - tamanho = constante
    - ponto = vento muito fraco/calmaria
    """

    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.35
    )

    if wind_day.empty:

        ax.text(
            0.5,
            0.5,
            'Sem dados de vento',
            transform=ax.transAxes,
            ha='center',
            va='center'
        )

        ax.set_yticks([])
        ax.set_ylabel('Vento')
        return

    dfw = wind_day.copy()

    # uma seta a cada 2 horas por padrão
    dfw = dfw.iloc[::passo]

    for timestamp, row in dfw.iterrows():

        ws = row['ws']
        wd = row['wd_deg']

        if pd.isna(ws):
            continue

        # --------------------------------------------------
        # Vento muito fraco → ponto
        # --------------------------------------------------

        if ws <= calm_threshold:

            ax.scatter(
                timestamp,
                0,
                s=35,
                color=cmap(norm(ws)),
                edgecolors='black',
                linewidths=0.4,
                zorder=3
            )

            continue

        # sem direção válida → não desenha seta
        if pd.isna(wd):
            continue

        # --------------------------------------------------
        # METEOROLOGIA:
        #
        # wd = DE ONDE o vento vem
        #
        # Nós queremos que a seta mostre PARA ONDE
        # a massa de ar está indo.
        #
        # wd=0   (N) → fluxo para Sul
        # wd=90  (E) → fluxo para Oeste
        # wd=180 (S) → fluxo para Norte
        # wd=270 (W) → fluxo para Leste
        # --------------------------------------------------

        angle_plot = (270 - wd) % 360

        marker_rotated = (
            Affine2D()
            .rotate_deg(angle_plot)
            .transform_path(ARROW_MARKER)
        )

        ax.scatter(
            timestamp,
            0,
            marker=marker_rotated,
            s=marker_size,
            color=cmap(norm(ws)),
            edgecolors='none',
            zorder=3
        )

    ax.set_ylim(
        -0.65,
        0.65
    )

    ax.set_yticks([])

    ax.set_ylabel(
        'Vento'
    )

    ax.grid(
        axis='x',
        alpha=0.15
    )
    
    
# ==========================================================
# TRIAGEM VISUAL - GERAR GRÁFICOS DE TODOS OS DIAS
# ==========================================================

pasta_triagem = "E:/totais/triagem_npf"

os.makedirs(
    pasta_triagem,
    exist_ok=True
)


# ----------------------------------------------------------
# Organizar as modas por DATA
# Isso é mais seguro do que assumir que o índice i bate
# perfeitamente entre duas listas.
# ----------------------------------------------------------

modas_por_data = {
    df_dia.index[0].date(): df_dia
    for df_dia in dfs_por_dia_modas_todos
    if not df_dia.empty
}

def onselect(xmin, xmax):
    # Blindagem contra o bug de duplo disparo do Matplotlib
    if xmin == xmax:
        return
        
    global text_boxes, delta_t_box
    global x_num, i
    global df, fig, ax
    global dp_peak_global_dia  

    start_dt = (
        mdates.num2date(xmin)
        .replace(tzinfo=None)
    )

    end_dt = (
        mdates.num2date(xmax)
        .replace(tzinfo=None)
    )

    print(f'Intervalo selecionado: {start_dt} até {end_dt}')

    # Remove retas MC e pontos anteriores
    for line in list(ax[0].lines):
        if str(line.get_label()).startswith('_GR_MC'):
            line.remove()
            
    for collection in list(ax[0].collections):
        if str(collection.get_label()).startswith('_PONTOS_MC'):
            collection.remove()

    for tb in text_boxes:
        tb.remove()

    text_boxes.clear()

    if delta_t_box is not None:
        delta_t_box.remove()
        delta_t_box = None

    # =====================================================
    # GR PELO Dpg (COM ALGORITMO INTELIGENTE DE CORTE)
    # Aqui a onselect "chama" o truncar_teto!
    # =====================================================
    
    resultado_nuc = calcular_gr_dpg(dpg_nuc, start_dt, end_dt, 10.6, 25.0, truncar_teto=True)
    resultado_aitken = calcular_gr_dpg(dpg_aitken, start_dt, end_dt, 25.0, 100.0, truncar_teto=True)
    resultado_acc = calcular_gr_dpg(dpg_acc, start_dt, end_dt, 100.0, 429.4, truncar_teto=True)
    
    # O Global não trunca, rastreia a banana inteira!
    resultado_global = calcular_gr_dpg(dp_peak_global_dia, start_dt, end_dt, 10.6, 600.0, truncar_teto=False)

    # =====================================================
    # DESENHAR OS AJUSTES MC (INCLUINDO O GLOBAL)
    # =====================================================
    
    # 1) NUCLEAÇÃO (Cyan)
    if resultado_nuc is not None:
        pontos_nuc = resultado_nuc['pontos']
        ax[0].scatter(pontos_nuc.index, pontos_nuc.values, color='cyan', marker='x', s=55, linewidths=2, zorder=32, label='_PONTOS_MC_NUC')
        ax[0].plot(resultado_nuc['t_linha'], resultado_nuc['dp_linha'], '--', color='cyan', linewidth=2.5, zorder=31, label='_GR_MC_NUC')
    
    # 2) AITKEN (Limegreen)
    if resultado_aitken is not None:
        pontos_aitken = resultado_aitken['pontos']
        ax[0].scatter(pontos_aitken.index, pontos_aitken.values, color='limegreen', marker='x', s=55, linewidths=2, zorder=32, label='_PONTOS_MC_AITKEN')
        ax[0].plot(resultado_aitken['t_linha'], resultado_aitken['dp_linha'], '--', color='limegreen', linewidth=2.5, zorder=31, label='_GR_MC_AITKEN')
    
    # 3) ACUMULAÇÃO (Black)
    if resultado_acc is not None:
        pontos_acc = resultado_acc['pontos']
        ax[0].scatter(pontos_acc.index, pontos_acc.values, color='black', marker='x', s=55, linewidths=2, zorder=32, label='_PONTOS_MC_ACC')
        ax[0].plot(resultado_acc['t_linha'], resultado_acc['dp_linha'], '--', color='black', linewidth=2.5, zorder=31, label='_GR_MC_ACC')

    # 4) PICO GLOBAL (Roxo)
    if resultado_global is not None:
        pontos_global = resultado_global['pontos']
        ax[0].scatter(pontos_global.index, pontos_global.values, color='#9400D3', marker='o', s=60, alpha=0.9, edgecolors='white', linewidths=1, zorder=35, label='_PONTOS_MC_GLOB')
        ax[0].plot(resultado_global['t_linha'], resultado_global['dp_linha'], '-', color='#9400D3', linewidth=3.5, zorder=34, label='_GR_MC_GLOB')

    # =====================================================
    # INFORMAÇÕES DOS 4 AJUSTES Dpg NA TELA
    # =====================================================
    linhas_gr = []
    
    if resultado_nuc is not None:
        linhas_gr.append(f'Nuc (10-25): {resultado_nuc["GR"]:.2f} nm/h  |  R² = {resultado_nuc["R2"]:.2f}')
    if resultado_aitken is not None:
        linhas_gr.append(f'Ait (25-100): {resultado_aitken["GR"]:.2f} nm/h  |  R² = {resultado_aitken["R2"]:.2f}')
    if resultado_acc is not None:
        linhas_gr.append(f'Acc (100-429): {resultado_acc["GR"]:.2f} nm/h  |  R² = {resultado_acc["R2"]:.2f}')
    if resultado_global is not None:
        linhas_gr.append(f'GLOBAL (Pico): {resultado_global["GR"]:.2f} nm/h  |  R² = {resultado_global["R2"]:.2f}')
    
    texto_gr = '\n'.join(linhas_gr)
    
    text_boxes.append(
        fig.text(
            0.58, 0.975, texto_gr,
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.5')
        )
    )

    delta_t_minutes = (end_dt - start_dt).total_seconds() / 60
    delta_t_box = fig.text(
        0.80, 0.95, f'Delta T: {delta_t_minutes:.1f} min',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    # =====================================================
    # REGISTRAR SELEÇÃO NO DICIONÁRIO E EXTRAIR DELTA T DA MODA
    # =====================================================
    gr_info = {
        'GR_Global': round(resultado_global['GR'], 2) if resultado_global else np.nan,
        
        'GR_10_25': round(resultado_nuc['GR'], 2) if resultado_nuc else np.nan,
        'Start_Nuc': resultado_nuc['pontos'].index.min().strftime('%H:%M:%S') if resultado_nuc else None,
        'End_Nuc': resultado_nuc['pontos'].index.max().strftime('%H:%M:%S') if resultado_nuc else None,
        
        'GR_25_100': round(resultado_aitken['GR'], 2) if resultado_aitken else np.nan,
        'Start_Aitken': resultado_aitken['pontos'].index.min().strftime('%H:%M:%S') if resultado_aitken else None,
        'End_Aitken': resultado_aitken['pontos'].index.max().strftime('%H:%M:%S') if resultado_aitken else None
    }
    
    novo_dado = (start_dt, end_dt, gr_info, None)
    
    if selected_data_list and selected_data_list[-1][3] is None:
        selected_data_list[-1] = novo_dado
    else:
        selected_data_list.append(novo_dado)
        
    fig.canvas.draw_idle()


# ==============================================================================
# LOOP PRINCIPAL DA TRIAGEM INTERATIVA (O DIA A DIA)
# ==============================================================================
i = 0
while i < len(dfs_por_dia) and not stop_flag:

    df = dfs_por_dia[i]
    df2 = dfs_por_dia[i]
    
    # Calcula o Pico Global para o dia inteiro (Disponibiliza para o onselect)
    global dp_peak_global_dia
    dp_peak_global_dia = calcular_dp_peak_global(df)
    
    # =====================================================
    # DIA E TRAÇADORES DE DIÂMETRO
    # =====================================================
    data_obj = df.index[0].date()
    data = df.index[0].strftime('%Y-%m-%d')
    
    # =====================================================
    # Dpg DAS MODAS MATLAB DO DIA ATUAL
    # =====================================================
    moda_dia = modas_por_data.get(data_obj, pd.DataFrame()).copy()
    
    if not moda_dia.empty:
        moda_dia.index = pd.to_datetime(moda_dia.index)
        if moda_dia.index.tz is not None:
            moda_dia.index = moda_dia.index.tz_localize(None)
    
        # Nucleação
        dpg_nuc = pd.to_numeric(moda_dia['Diâmetro geométricos médio Nucleação'], errors='coerce')
        valido_nuc = dpg_nuc.notna() & (dpg_nuc >= 10.6) & (dpg_nuc < 25)
    
        # Aitken
        dpg_aitken = pd.to_numeric(moda_dia['Diâmetro geométricos médio Aitken'], errors='coerce')
        valido_aitken = dpg_aitken.notna() & (dpg_aitken >= 25) & (dpg_aitken < 100)
    
        # Acumulação
        dpg_acc = pd.to_numeric(moda_dia['Diâmetro geométricos médio Acumulação'], errors='coerce')
        valido_acc = dpg_acc.notna() & (dpg_acc >= 100) & (dpg_acc <= 429.4)
        
    dia_inicio = df.index[0].normalize()
    dia_fim = dia_inicio + pd.Timedelta(days=1)

    # =====================================================
    # PNC, RADIAÇÃO, O3 E VENTO
    # =====================================================
    N_total = integrar_pnc(df, 10.6, 429.4)
    N_10_25 = integrar_pnc(df, 10.6, 25)
    df_total_conc = pd.DataFrame({'N_total': N_total, 'N_10_25': N_10_25})

    irr_dia = irradiancia.loc[(irradiancia.index >= dia_inicio) & (irradiancia.index < dia_fim)]
    o3_dia = serie_o3.loc[(serie_o3.index >= dia_inicio) & (serie_o3.index < dia_fim)]
    vento_dia = wind_hour.loc[(wind_hour.index >= dia_inicio) & (wind_hour.index < dia_fim)].copy()

    # =====================================================
    # PAINEL PRINCIPAL INTERATIVO (PUBLICATION QUALITY)
    # =====================================================
    fig, ax = plt.subplots(
        5, 1, 
        figsize=(18, 14), # Tamanho ajustado para caber perfeito no monitor
        sharex=True,
        gridspec_kw={'height_ratios': [4, 1.2, 1, 1, 0.8]}
    )

    # ==========================================================
    # CONTORNO PNSD
    # ==========================================================
    df_conc_filtrado = df2.T
    x_num = mdates.date2num(df.index.to_pydatetime())
    y_vals = pd.to_numeric(df_conc_filtrado.index, errors='coerce')
    X, Y = np.meshgrid(x_num, y_vals)
    Z = df_conc_filtrado.values
    masked_Z = np.ma.masked_where(Z <= 1, Z)
    
    p = ax[0].pcolormesh(X, Y, masked_Z, norm=colors.LogNorm(vmin=1e1, vmax=5e4), cmap='nipy_spectral', shading='auto')
    ax[0].set_yscale('log')
    
    y_ticks = [10.6, 20, 50, 100, 200, 300, 400, 500, 600]
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(['10.6', '20', '50', '100', '200', '300', '400', '500', '600'])
    ax[0].set_ylim(10.6, 600)
    
    # Linhas divisórias (25 nm e 100 nm)
    ax[0].axhline(y=25, color='white', linestyle='-', linewidth=2, alpha=0.9, zorder=10)
    ax[0].axhline(y=100, color='white', linestyle='--', linewidth=1.5, alpha=0.75, zorder=10)
    
    ax[0].set_ylabel('Particle Diameter, Dp (nm)', fontweight='bold')
    ax[0].set_title(f'PNSD Dynamics - {data}', fontweight='bold', fontsize=16)
    
    # ==========================================================
    # Dpg DAS MODAS MATLAB SOBRE O CONTORNO
    # ==========================================================
    if not moda_dia.empty:
        ax[0].scatter(moda_dia.index[valido_nuc], dpg_nuc[valido_nuc], color='blue', s=18, alpha=0.90, zorder=25, label='Dpg — nucleação')
        ax[0].scatter(moda_dia.index[valido_aitken], dpg_aitken[valido_aitken], color='limegreen', s=18, alpha=0.90, zorder=25, label='Dpg — Aitken')
        ax[0].scatter(moda_dia.index[valido_acc], dpg_acc[valido_acc], color='black', edgecolors='white', linewidths=0.4, s=20, alpha=0.85, zorder=25, label='Dpg — acumulação')

    # ==========================================================
    # GRÁFICOS INFERIORES (PNC, RAD, O3)
    # ==========================================================
    # PNC
    ax[1].plot(df_total_conc.index, df_total_conc['N_total'], linewidth=2.0, label='N total', color='#333333')
    ax[1].plot(df_total_conc.index, df_total_conc['N_10_25'], linewidth=2.0, label='N 10.6–25 nm', color='#D95319')
    ax[1].set_ylabel('PNC\n(# cm$^{-3}$)')
    ax[1].set_ylim(bottom=0)
    ax[1].legend(loc='upper right', ncol=2, frameon=False)
    ax[1].yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Radiação
    if not irr_dia.empty:
        ax[2].plot(irr_dia.index, irr_dia.values, linewidth=2.0, color='#EDB120')
        ax[2].fill_between(irr_dia.index, 0, irr_dia.values, alpha=0.25, color='#EDB120')
    ax[2].set_ylabel('Radiation\n(W m$^{-2}$)')
    ax[2].set_ylim(bottom=0)
    ax[2].yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')

    # Ozônio
    if not o3_dia.empty:
        ax[3].plot(o3_dia.index, o3_dia.values, linewidth=2.0, color='#0072BD')
    ax[3].set_ylabel('O$_3$')
    ax[3].set_ylim(bottom=0)
    ax[3].yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')

    # ==========================================================
    # VENTO E EIXOS TEMPORAIS
    # ==========================================================
    plot_wind_panel(ax[4], vento_dia, norm=wind_norm, cmap=wind_cmap, calm_threshold=0.3, passo=2, marker_size=260)
    ax[4].set_xlabel('Local Time (h)', fontweight='bold')
    
    sm_wind = ScalarMappable(norm=wind_norm, cmap=wind_cmap)
    sm_wind.set_array([])
    cbar_wind = fig.colorbar(sm_wind, ax=ax[4], orientation='vertical', pad=0.01, fraction=0.015)
    cbar_wind.set_label('Wind speed (m s$^{-1}$)', fontsize=11)
    
    for a in ax:
        a.set_xlim(dia_inicio, dia_fim)
        # Fechar caixas
        for spine in a.spines.values():
            spine.set_visible(True)

    ax[4].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # ==========================================================
    # SPAN SELECTOR E INTERFACES
    # ==========================================================
    span = SpanSelector(
        ax[0], onselect, 'horizontal', useblit=True,
        props=dict(alpha=0.5, facecolor='red')
    )

    # Imagem (Opcional - mantenha o caminho correto do seu PC)
    image_path = 'C:\\Users\\thela\\maskNPF\\esquema.png'
    try:
        image = plt.imread(image_path)
        imagebox = OffsetImage(image, zoom=0.4)
        ab = AnnotationBbox(imagebox, (0.28, 0.85), xycoords='axes fraction', frameon=False)
        ax[0].add_artist(ab)
    except FileNotFoundError:
        pass

    # BOTÕES
    button_class_I_ax = plt.axes([0.02, 0.90, 0.12, 0.03])
    button_class_I = CustomButton(button_class_I_ax, 'Classe I')
    button_class_I.on_clicked(lambda event: classify_data('Classe I', button_class_I))

    button_class_Ib_ax = plt.axes([0.02, 0.87, 0.12, 0.03])
    button_class_Ib = CustomButton(button_class_Ib_ax, 'Classe Ib')
    button_class_Ib.on_clicked(lambda event: classify_data('Classe Ib', button_class_Ib))

    button_class_II_ax = plt.axes([0.02, 0.84, 0.12, 0.03])
    button_class_II = CustomButton(button_class_II_ax, 'Classe II')
    button_class_II.on_clicked(lambda event: classify_data('Classe II', button_class_II))

    button_class_ind_ax = plt.axes([0.02, 0.81, 0.12, 0.03])
    button_class_ind = CustomButton(button_class_ind_ax, 'Indeterminado')
    button_class_ind.on_clicked(lambda event: classify_data('Indeterminado', button_class_ind))

    button_next_no_save_ax = plt.axes([0.02, 0.78, 0.12, 0.03])
    button_next_no_save = CustomButton(button_next_no_save_ax, 'Próximo-sem salvar')
    button_next_no_save.on_clicked(next_without_saving)

    button_next_save_ax = plt.axes([0.02, 0.75, 0.12, 0.03])
    button_next_save = CustomButton(button_next_save_ax, 'Próximo-salvar')
    button_next_save.on_clicked(next_with_saving)

    button_stop_ax = plt.axes([0.02, 0.72, 0.12, 0.03])
    button_stop = CustomButton(button_stop_ax, 'Parar')
    button_stop.on_clicked(stop_annotation)

    button_toggle_img_ax = plt.axes([0.02, 0.69, 0.12, 0.03])
    button_toggle_img = CustomButton(button_toggle_img_ax, 'Mostrar/Ocultar imagem')
    button_toggle_img.on_clicked(toggle_image_visibility)


    button_save_evento_ax = plt.axes([0.02, 0.64, 0.12, 0.04]) # Botão mais alto
    button_save_evento = CustomButton(button_save_evento_ax, 'Salvar Evento Completo', color='lightblue')
    button_save_evento.on_clicked(save_evento_completo)

    class_buttons = [button_class_I, button_class_Ib, button_class_II, button_class_ind]
    class_buttons = [button_class_I, button_class_Ib, button_class_II, button_class_ind]

    fig.subplots_adjust(left=0.17, right=0.91, top=0.94, bottom=0.06, hspace=0.10)
    plt.show(block=True)
    
    i += 1  # Move to the next dataframe only if not stopped

save_selected_data()


dfs_por_dia_filtrado_integral = []



df_filtrado_integral = df_ext.loc[:, (df_ext.columns.astype(float) >= 9) & (df_ext.columns.astype(float) <= 100)]



for day, df_day in df_filtrado_integral.groupby(df_filtrado_integral.index.date):

    dfs_por_dia_filtrado_integral.append(df_day)

dfs_por_dia_filtrado_integral = [df for df in dfs_por_dia_filtrado_integral if pd.Series(df.index.date).isin(dates_to_filter).any()]

selected_data_list = []



current_class = None

stop_flag = False

text_boxes = []

delta_t_box = None


for numero, df in enumerate(dfs_por_dia_todos, start=1):

    if df.empty:
        continue

    data_obj = df.index[0].date()
    data_str = df.index[0].strftime('%Y-%m-%d')

    print(
        f"[{numero}/{len(dfs_por_dia)}] "
        f"Gerando {data_str}"
    )

    # ======================================================
    # PERÍODO DO DIA
    # ======================================================

    dia_inicio = pd.Timestamp(data_obj)
    dia_fim = dia_inicio + pd.Timedelta(days=1)

    # ======================================================
    # Dpg DAS MODAS AJUSTADAS NO MATLAB
    # ======================================================
    
    moda_dia = modas_por_data.get(
        data_obj,
        pd.DataFrame()
    ).copy()
    
    if not moda_dia.empty:
    
        # garantir índice datetime sem timezone
        moda_dia.index = pd.to_datetime(
            moda_dia.index
        )
    
        if moda_dia.index.tz is not None:
            moda_dia.index = moda_dia.index.tz_localize(None)
    
        # --------------------------------------------------
        # Nucleação
        # --------------------------------------------------
        dpg_nuc = pd.to_numeric(
            moda_dia['Diâmetro geométricos médio Nucleação'],
            errors='coerce'
        )
    
        valido_nuc = (
            dpg_nuc.notna() &
            (dpg_nuc >= 10.6) &
            (dpg_nuc < 25)
        )
    
        # --------------------------------------------------
        # Aitken
        # --------------------------------------------------
        dpg_aitken = pd.to_numeric(
            moda_dia['Diâmetro geométricos médio Aitken'],
            errors='coerce'
        )
    
        valido_aitken = (
            dpg_aitken.notna() &
            (dpg_aitken >= 25) &
            (dpg_aitken < 100)
        )
    
        # --------------------------------------------------
        # Acumulação
        # --------------------------------------------------
        dpg_acc = pd.to_numeric(
            moda_dia['Diâmetro geométricos médio Acumulação'],
            errors='coerce'
        )
    
        valido_acc = (
            dpg_acc.notna() &
            (dpg_acc >= 100) &
            (dpg_acc <= 429.4)
        )
        
    
    
        # ======================================================
        # CONCENTRAÇÕES
        # ======================================================
    
        N_total = integrar_pnc(
            df,
            10.6,
            429.4
        )
    
        N_10_25 = integrar_pnc(
            df,
            10.6,
            25
        )
    
        df_pnc = pd.DataFrame({
            "N_total": N_total,
            "N_10_25": N_10_25
        })

    # ======================================================
    # IRRADIÂNCIA
    # ======================================================

    irr_dia = irradiancia.loc[
        (irradiancia.index >= dia_inicio) &
        (irradiancia.index < dia_fim)
    ]

    # ======================================================
    # OZÔNIO
    # ======================================================

    o3_dia = serie_o3.loc[
        (serie_o3.index >= dia_inicio) &
        (serie_o3.index < dia_fim)
    ]
    # ======================================================
    # VENTO
    # ======================================================
    
    vento_dia = wind_hour.loc[
        (wind_hour.index >= dia_inicio) &
        (wind_hour.index < dia_fim)
    ].copy()

    # ======================================================
    # FIGURA
    # ======================================================

    fig, ax = plt.subplots(
    5,
    1,
    figsize=(18, 13),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            4,
            1.2,
            1,
            1,
            0.8
        ]
    }
)

    # ======================================================
    # 1 - CONTORNO PNSD
    # ======================================================

    df_plot = df.T

    x_num = mdates.date2num(
        df.index.to_pydatetime()
    )

    y_vals = pd.to_numeric(
        df_plot.index,
        errors="coerce"
    )

    X, Y = np.meshgrid(
        x_num,
        y_vals
    )

    Z = df_plot.values

    masked_Z = np.ma.masked_where(
        Z <= 1,
        Z
    )

    p = ax[0].pcolormesh(
        X,
        Y,
        masked_Z,
        norm=colors.LogNorm(
            vmin=1e1,
            vmax=5e4
        ),
        cmap="nipy_spectral",
        shading="auto"
    )

    ax[0].set_yscale("log")

    y_ticks = [
    10.6, 20, 50, 100,
    200, 300, 400, 500, 600
    ]
    
    ax[0].set_yticks(y_ticks)
    
    ax[0].set_yticklabels([
        '10.6', '20', '50', '100',
        '200', '300', '400', '500', '600'
    ])

    ax[0].set_ylim(10.6, 600)

    ax[0].set_yticks(y_ticks)

    ax[0].set_yticklabels(
        [str(x) for x in y_ticks]
    )

    # linha de 25 nm
    ax[0].axhline(
        25,
        color="white",
        linewidth=1.5,
        alpha=0.9
    )
    ax[0].axhline(
    100,
    color='white',
    linestyle='--',
    linewidth=1.3,
    alpha=0.75,
    zorder=10
    )

        # ======================================================
    # Dpg DAS MODAS MATLAB SOBRE O CONTORNO
    # ======================================================
    
    if not moda_dia.empty:
    
        # Nucleação
        ax[0].scatter(
            moda_dia.index[valido_nuc],
            dpg_nuc[valido_nuc],
            color='blue',
            s=14,
            alpha=0.85,
            zorder=20,
            label='Dpg — nucleação'
        )
    
        # Aitken
        ax[0].scatter(
            moda_dia.index[valido_aitken],
            dpg_aitken[valido_aitken],
            color='limegreen',
            s=14,
            alpha=0.85,
            zorder=20,
            label='Dpg — Aitken'
        )
    
        # Acumulação
        ax[0].scatter(
            moda_dia.index[valido_acc],
            dpg_acc[valido_acc],
            color='black',
            edgecolors='white',
            linewidths=0.3,
            s=16,
            alpha=0.80,
            zorder=20,
            label='Dpg — acumulação'
        )

    # ======================================================
    # 2 - PNC
    # ======================================================

    ax[1].plot(
        df_pnc.index,
        df_pnc["N_total"],
        linewidth=1.3,
        label="N total"
    )

    ax[1].plot(
        df_pnc.index,
        df_pnc["N_10_25"],
        linewidth=1.3,
        label="N 10.6–25 nm"
    )

    ax[1].set_ylabel(
        "PNC\n(# cm$^{-3}$)"
    )

    ax[1].set_ylim(
        bottom=0
    )

    ax[1].legend(
        loc="upper right",
        ncol=2,
        fontsize=9
    )

    ax[1].grid(
        alpha=0.2
    )

    # ======================================================
    # 3 - RADIAÇÃO
    # ======================================================

    if not irr_dia.empty:

        ax[2].plot(
            irr_dia.index,
            irr_dia.values,
            linewidth=1.3
        )

        ax[2].fill_between(
            irr_dia.index,
            0,
            irr_dia.values,
            alpha=0.15
        )

    ax[2].set_ylabel(
        "Radiação\n(W m$^{-2}$)"
    )

    ax[2].set_ylim(
        bottom=0
    )

    ax[2].grid(
        alpha=0.2
    )

    # ======================================================
    # 4 - OZÔNIO
    # ======================================================

    if not o3_dia.empty:

        ax[3].plot(
            o3_dia.index,
            o3_dia.values,
            linewidth=1.3
        )

    ax[3].set_ylabel(
        "O$_3$"
    )


    ax[3].set_ylim(
        bottom=0
    )

    ax[3].grid(
        alpha=0.2
    )

    # ======================================================
    # EIXO TEMPORAL COMUM
    # ======================================================

    for a in ax:

        a.set_xlim(
            dia_inicio,
            dia_fim
        )

    ax[4].xaxis.set_major_locator(
        mdates.HourLocator(interval=2)
    )
    
    ax[4].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )
    # ======================================================
    # 5 - VENTO
    # ======================================================
    
    plot_wind_panel(
        ax[4],
        vento_dia,
        norm=wind_norm,
        cmap=wind_cmap,
        calm_threshold=0.3,
        passo=2,
        marker_size=230
    )
    
    ax[4].set_xlabel(
        'Hora local'
    )
    
    sm_wind = ScalarMappable(
    norm=wind_norm,
    cmap=wind_cmap
    )
    
    sm_wind.set_array([])
    
    cbar_wind = fig.colorbar(
        sm_wind,
        ax=ax[4],
        orientation='vertical',
        pad=0.01,
        fraction=0.018
    )
    
    cbar_wind.set_label(
        'Vento (m s$^{-1}$)',
        fontsize=10
    )
    
    cbar_wind.ax.tick_params(
        labelsize=9
    )

    # ======================================================
    # COLORBAR
    # ======================================================

    cax_pnsd = ax[0].inset_axes(
        [1.01, 0.0, 0.012, 1.0]
    )
    
    cbar = fig.colorbar(
        p,
        cax=cax_pnsd
    )
    
    cbar.set_label(
        'dN/dlog$D_p$ (cm$^{-3}$)'
    )

    # ======================================================
    # SALVAR
    # ======================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.91,
        top=0.95,
        bottom=0.07,
        hspace=0.10
    )

    arquivo = os.path.join(
        pasta_triagem,
        f"{data_str}_diagnostico_NPF.png"
    )

    fig.savefig(
        arquivo,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


print(
    "\nTriagem concluída!"
)

print(
    "Gráficos salvos em:",
    pasta_triagem
)
# ==========================================================
# GERAR PDF COM TODAS AS IMAGENS DE TRIAGEM
# ==========================================================
print("\nGerando PDF com todas as imagens para análise...")

# Pega todos os caminhos dos arquivos PNG gerados na pasta de triagem
# O sorted() garante que as imagens ficarão na ordem cronológica (pelos nomes dos arquivos)
imagens_png = sorted(glob.glob(os.path.join(pasta_triagem, "*.png")))

if imagens_png:
    lista_imagens = []
    
    # Abre cada imagem e converte para o formato RGB (necessário para o PDF)
    for caminho_img in imagens_png:
        img = Image.open(caminho_img).convert('RGB')
        lista_imagens.append(img)
    
    # O arquivo PDF será salvo na mesma pasta
    caminho_pdf = os.path.join(pasta_triagem, "Relatorio_Triagem_NPF.pdf")
    
    # Salva a primeira imagem e anexa o restante das imagens como novas páginas
    primeira_imagem = lista_imagens[0]
    primeira_imagem.save(
        caminho_pdf,
        save_all=True,
        append_images=lista_imagens[1:]
    )
    
    print(f"Relatório PDF criado com sucesso em: {caminho_pdf}")
else:
    print("Nenhuma imagem encontrada para gerar o PDF.")
i = 0






##################################################################################



import pandas as pd
import numpy as np
import re

# ==========================================================
# CARREGAR E PREPARAR df_g CORRIGINDO O FORMATO DO CSV
# ==========================================================
csv_path = 'E:/totais/selected_data_fim.csv'

parsed_rows = []
with open(csv_path, 'r', encoding='unicode_escape') as f:
    lines = f.readlines()[1:] # Pula o cabeçalho
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Expressão regular para separar start_datetime, end_datetime, GRs e classe
        match = re.match(r'^([^,]+),([^,]+),(.*),([^,]+)$', line)
        if match:
            start_dt, end_dt, gr_dict_str, classe = match.groups()
            parsed_rows.append({
                'start_datetime': start_dt.strip('"'),
                'end_datetime': end_dt.strip('"'),
                'GRs': gr_dict_str.strip('"'),
                'classe': classe.strip('"')
            })

df_g = pd.DataFrame(parsed_rows)
df_g['start_datetime'] = pd.to_datetime(df_g['start_datetime']).dt.floor("T")  # Arredonda para o minuto
df_g['end_datetime'] = pd.to_datetime(df_g['end_datetime']).dt.floor("T")      # Arredonda para o minuto
df_g = df_g.dropna(subset=['start_datetime', 'end_datetime'])

print(f"Total de eventos carregados com sucesso: {len(df_g)}")
df_g = df_g[df_g['classe'].isin(['Classe I', 'Classe Ib'])].copy()
# Lista para armazenar os DataFrames filtrados
filtered_dfs = []

# Iterando em cada linha de df_g e no DataFrame correspondente em dfs_por_dia
for i in range(len(df_g)):
    if i < len(dfs_por_dia):  # Verifica se o índice está dentro do limite da lista
        row = df_g.iloc[i]
        start = row['start_datetime']
        end = row['end_datetime']
        
        # Obtendo o DataFrame correspondente
        df = dfs_por_dia[i]
        
        # Encontrando os índices mais próximos
        nearest_start_idx = df.index.get_indexer([start], method='nearest')[0]
        nearest_end_idx = df.index.get_indexer([end], method='nearest')[0]
        
        nearest_start = df.index[nearest_start_idx]
        nearest_end = df.index[nearest_end_idx]
        
        # Filtrando o DataFrame pelo índice datetime
        filtered_df = df.loc[nearest_start:nearest_end]
        filtered_dfs.append(filtered_df)


    
    
# Calcular CoagS para cada DataFrame
coagS_por_dia = [calculate_coags(df) for df in filtered_dfs]

# Calculate GR_scav for each DataFrame
gr_scav_por_dia = [calculate_gr_scav(df, coagS.sum(axis=1)) for df, coagS in zip(filtered_dfs, coagS_por_dia)]

# Imprimir os resultados
for i, gr_scav in enumerate(gr_scav_por_dia):
    print(f"Dia {i+1}: GR_scav (nm/h) = {gr_scav:.8f}")

# Iterar sobre cada DataFrame e calcular GR_coag
gr_coag_list = []

for df in dfs_por_dia:
    df_gmd_10_25 = calcular_gmd_intervalo(df)
    gr_coag_df = calcular_gr_coag(df_gmd_10_25)
    gr_coag_list.append(gr_coag_df)

# Calcular a média de GR_coag para cada DataFrame e imprimir
for i, gr_coag_df in enumerate(gr_coag_list):
    gr_coag_mean = gr_coag_df['GR_coag'].mean()
    print(f"Dia {i+1}: Média de GR_coag = {gr_coag_mean:.8f} nm/h")
    
##################################################################################

    
    



# Constantes físicas
k = 1.38e-23  # Constante de Boltzmann (J/K)
T = 298  # Temperatura (K)
eta = 1.81e-5  # Viscosidade do ar (Pa.s)
lambda_air = 68e-9  # Caminho livre médio do ar (m)

def calcular_C(d_p):
    Kn = lambda_air / d_p
    return (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn**2)

def diffusion_coefficient(d_p):
    C = calcular_C(d_p)
    return (k * T * C) / (3 * np.pi * eta * d_p)

def coagulation_kernel(d_p_i, d_p_j):
    D_i = diffusion_coefficient(d_p_i)
    D_j = diffusion_coefficient(d_p_j)
    Kn = (lambda_air / d_p_i + lambda_air / d_p_j) / 2
    beta = (1 + Kn) / (1 + 1.71 * Kn + 1.33 * Kn**2)  # Fator de correção de Fuchs-Sutugin
    return (D_i + D_j) * np.pi * beta

def calculate_coags(df):
    particle_diameter = pd.to_numeric(df.columns, errors='coerce')  # Converter para metros
    number_concentration = df.values

    coagS = []
    for i in range(len(particle_diameter)):
        d_p_i = particle_diameter[i]
        N_i = number_concentration[:, i]

        sum_coag = 0
        for j in range(len(particle_diameter)):
            d_p_j = particle_diameter[j]
            N_j = number_concentration[:, j]

            k_dpij = coagulation_kernel(d_p_i, d_p_j)
            sum_coag += k_dpij * N_j

        coagS.append(sum_coag * N_i.mean())  # Média da concentração numérica

    return np.array(coagS)
    
# Calcular CoagS para cada DataFrame
coagS_por_dia = [calculate_coags(df) for df in filtered_dfs]

# Calculate GR_scav for each DataFrame
gr_scav_por_dia = [calculate_gr_scav(df, coagS.sum(axis=1)) for df, coagS in zip(filtered_dfs, coagS_por_dia)]

# Imprimir os resultados
for i, gr_scav in enumerate(gr_scav_por_dia):
    print(f"Dia {i+1}: GR_scav (nm/h) = {gr_scav:.8f}")

# Iterar sobre cada DataFrame e calcular GR_coag
gr_coag_list = []

for df in dfs_por_dia:
    df_gmd_10_25 = calcular_gmd_intervalo(df)
    gr_coag_df = calcular_gr_coag(df_gmd_10_25)
    gr_coag_list.append(gr_coag_df)

# Calcular a média de GR_coag para cada DataFrame e imprimir
for i, gr_coag_df in enumerate(gr_coag_list):
    gr_coag_mean = gr_coag_df['GR_coag'].mean()
    print(f"Dia {i+1}: Média de GR_coag = {gr_coag_mean:.8f} nm/h")
    
    


# Constantes (substitua com seus valores)
lambda_air = 68e-9   # Caminho livre médio do vapor [m]

# Função para calcular o fator de correção C
def calcular_C(d):
    C = 1 + (lambda_air / d) * (2.514 + 0.8 * np.exp(-0.55 * d / lambda_air))
    return C

# Função para calcular k(dp)
def calcular_k_dp(d):
    C = calcular_C(d)
    return 3e-16 * C

# Função para calcular a concentração (N) de cada partícula para o intervalo de 9 a 25 nm
def calcular_concentracao(df, lower_bound=9, upper_bound=600):
    # Converter os diâmetros das colunas para valores numéricos
    logDp_values = pd.to_numeric(df.columns, errors='coerce')
    
    # Filtrar os diâmetros entre 9 e 25 nm
    mask = (logDp_values >= lower_bound) & (logDp_values <= upper_bound)
    filtered_logDp_values = logDp_values[mask].values  # Converter para numpy array
    
    # Lista para armazenar as concentrações de cada linha
    concentracao = []
    
    # Iterar sobre as linhas do DataFrame (cada linha é um ponto no tempo)
    for index, row in df.iterrows():
        # Filtrar as concentrações para o intervalo de diâmetro
        mean_concentration = row[mask]
        
        if mean_concentration.isnull().any():
            concentracao.append([np.nan] * len(filtered_logDp_values))
            continue
        
        # Calcular dlog(dp) (diferença logarítmica)
        dlogDp = np.diff(np.log10(filtered_logDp_values))  # Diferença logarítmica entre diâmetros
        
        # Calcular a concentração (N) para cada partícula
        N = mean_concentration.values[:-1] * dlogDp  # Concentração para cada diâmetro
        concentracao.append(N)
    
    # Criar o DataFrame de concentrações (com datetime no índice e diâmetros nas colunas)
    conc_df = pd.DataFrame(concentracao, columns=filtered_logDp_values[:-1], index=df.index)
    return conc_df

# Supondo que filtered_dfs seja a lista de DataFrames
filtered_dfs_conc = [calcular_concentracao(df) for df in filtered_dfs]

# Exemplo de como acessar o resultado
for i, conc_dia in enumerate(filtered_dfs_conc):
    print(f"Dia {i+1}:")
    print(conc_dia.head())  # Exibe as primeiras linhas do DataFrame de concentração para cada dia
    
    
import numpy as np
import pandas as pd

# Constantes (substitua com seus valores!)
lambda_v = 68e-9   # Caminho livre médio do vapor [m]
alpha = 1.0        # Coeficiente de adesão (definido como 1)

# Constantes físicas
k = 1.38e-23  # Constante de Boltzmann (J/K)
T = 298  # Temperatura (K)
eta = 1.81e-5  # Viscosidade do ar (Pa.s)

# Função para calcular o coeficiente de difusão
def diffusion_coefficient(d_p):
    # Convertendo o diâmetro de nm para metros
    d_p_m = d_p * 1e-9  # Conversão de nm para metros
    C = (1 + lambda_v / d_p_m)
    return (k * T * C) / (3 * np.pi * eta * d_p_m)

# Função para calcular CS para um DataFrame
def calculate_cs(df):
    # Extrair diâmetros das colunas e converter para raio
    diameters = df.columns.astype(float)  # Os diâmetros estão em nm
    radii = diameters / 2  # Converte diâmetro para raio (ajuste se já for raio)
    
    # Calcular D para cada diâmetro individualmente
    D = np.array([diffusion_coefficient(d) for d in diameters])  # Calculando D para cada diâmetro
    
    # Calcular Kn e beta_M para cada tamanho de partícula
    radii_m = radii * 1e-9  # Conversão de nm para metros (raio)
    kn = lambda_v / radii_m
    beta_M = (kn + 1) / (((0.377 * kn) + 1) + ((4/3)*alpha**-1) * kn**2 + ((4/3)*alpha**-1) * kn)
    
    # Verificar se o comprimento de 'beta_M' é igual ao de 'radii'
    if len(beta_M) != len(radii):
        raise ValueError(f"Dimensões incompatíveis: beta_M tem {len(beta_M)} elementos, enquanto radii tem {len(radii)} elementos.")
    
    # Calcular CS' para cada linha (datetime) e garantir as dimensões corretas
    cs_values = []
    for row in df.itertuples(index=False):
        # Multiplicação elemento por elemento para cada linha
        cs_prime = np.sum(beta_M * radii * np.array(row))
        
        # Calcular CS = 4 * pi * D * CS'
        cs = 4 * np.pi * D * cs_prime  # Aqui, a multiplicação é feita corretamente
        
        # Aqui a unidade já está em partículas/cm³*s, pois np.array(row) já está em partículas/cm³
        # Multiplicamos por 10^-2 para o ajuste solicitado
        cs = cs 
        cs_values.append(cs)

    # Retornar o valor médio do CS para aquele dia
    return np.mean(cs_values, axis=0)

# Calcular CS para cada DataFrame em filtered_dfs
cs_por_dia = [calculate_cs(df) for df in filtered_dfs_conc]

# Criar uma lista de DataFrames com os valores médios do CS
cs_dfs = [pd.DataFrame(cs, columns=['CS']) for cs in cs_por_dia]

# Imprimir os resultados (valor médio de CS por dia)
for i, cs_df in enumerate(cs_dfs):
    cs_mean = cs_df['CS'].sum()  # Média do CS para aquele dia
    # Exibindo o valor em notação científica
    print(f"Dia {i+1}: Média de CS = {cs_mean:.2e} partículas/cm³*s")
    


# Filtrar as colunas com diâmetros entre 9 e 25
def filter_columns_by_diameter(df, min_diameter=9, max_diameter=600):
    # Converter os nomes das colunas para float, se necessário
    df.columns = df.columns.astype(float)
    
    # Filtrar as colunas com diâmetros entre min_diameter e max_diameter
    df_filtered = df.loc[:, (df.columns >= min_diameter) & (df.columns <= max_diameter)]
    
    return df_filtered

# Aplicar o filtro para cada DataFrame em filtered_dfs_conc
filtered_dfs_conc_filtered = [filter_columns_by_diameter(df) for df in filtered_dfs_conc]

# Verificar os resultados
for i, df in enumerate(filtered_dfs_conc_filtered):
    print(f"DataFrame {i+1} filtrado:\n", df.head())

import numpy as np
import pandas as pd

# Constantes físicas (ajuste conforme seus dados!)
k = 1.38e-23          # Constante de Boltzmann [J/K]
T = 298               # Temperatura [K]
mu = 1.81e-5          # Viscosidade dinâmica do ar [Pa·s]
densidade = 1400      # Densidade da partícula [kg/m³]
lambda_ar = 66e-9     # Caminho livre médio no ar [m]
a1, a2, a3 = 1.142, 0.558, 0.999  # Parâmetros de Cunningham

# Funções auxiliares
def cunningham_correction(Kn):
    return 1 + Kn * (a1 + a2 * np.exp(-a3 / Kn))

def particle_diffusion(R, Cc):
    return (k * T * Cc) / (6 * np.pi * mu * R)

def particle_mass(R):
    return (4/3) * np.pi * (R**3) * densidade

def thermal_velocity(m):
    return np.sqrt(8 * k * T / (np.pi * m))

def calculate_coags(df):
    # Extrair diâmetros e converter para raio (ajuste unidades se necessário)
    diameters = df.columns.astype(float)
    radii = diameters / 2  # Diâmetro → raio
    
    # Calcular Kn, Cc, D_i, m_i, c_bar_i para cada raio
    Kn = lambda_ar / radii
    Cc = cunningham_correction(Kn)
    D = particle_diffusion(radii, Cc)
    mass = particle_mass(radii)
    c_bar = thermal_velocity(mass)
    
    # Matriz de coeficientes K_ij para todas as combinações (i, j)
    n = len(radii)
    K_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            R1, R2 = radii[i], radii[j]
            D1, D2 = D[i], D[j]
            m1, m2 = mass[i], mass[j]
            
            # Calcular termos intermediários
            R12 = R1 + R2
            D12 = D1 + D2
            c_bar_12 = np.sqrt(c_bar[i]**2 + c_bar[j]**2)
            Kc_cont = 4 * np.pi * R12 * D12  # Termo do regime contínuo
            
            # Coeficiente de coagulação K_ij
            denominator = R12 + (4 * D12) / (c_bar_12 * R12)
            K_ij = Kc_cont / denominator
            K_matrix[i, j] = K_ij
    
    # Calcular CoagS para cada linha (datetime) de forma escalar
    coagS = df.apply(lambda row: np.sum(np.array(K_matrix) * row.values), axis=1)
    
    return coagS

# Calcular CoagS para cada DataFrame em filtered_dfs
coag_sinks_por_dia = [calculate_coags(df) for df in filtered_dfs_conc]

# Criar uma lista de DataFrames com os valores de CoagS
coag_sinks_dfs = [pd.DataFrame(coag, columns=['CoagS']) for coag in coag_sinks_por_dia]

# Imprimir os resultados (coeficiente de CoagS para cada dia)
for i, coag_df in enumerate(coag_sinks_dfs):
    coag_mean = coag_df['CoagS'].sum()  # Soma de CoagS para aquele dia
    print(f"Dia {i+1}: Média de CoagS = {coag_mean:.8e} partículas/cm³*s")





#######################################################################################
#######################################################################################

import numpy as np

# Constantes físicas
kB = 1.38e-23        # J/K
T = 298.0            # K
mu = 1.81e-5         # Pa s
lambda_air = 66e-9   # m
alpha = 1.0          # sticking coefficient

def cunningham(dp):
    Kn = lambda_air / dp
    return 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))

def diffusion_vapor():
    # Difusão típica do H2SO4 no ar (Kulmala 2004)
    return 0.06e-4  # m²/s

def beta_fuchs(dp):
    Kn = lambda_air / dp
    return (1 + Kn) / (1 + 1.71*Kn + 1.33*Kn**2)

def calculate_CS(df):
    """
    df: DataFrame diário
        colunas = dp (nm)
        valores = #/cm³
    retorna: Series CS(t) em s⁻¹
    """
    dp_nm = df.columns.astype(float).values
    dp_m = dp_nm * 1e-9

    Dv = diffusion_vapor()
    beta = beta_fuchs(dp_m)

    CS_list = []

    for _, row in df.iterrows():
        Ni = row.values * 1e6  # cm⁻³ → m⁻³
        CS = 2 * np.pi * Dv * np.sum(beta * dp_m * Ni)
        CS_list.append(CS)

    return pd.Series(CS_list, index=df.index)
cs_por_dia = [calculate_CS(df).mean() for df in filtered_dfs]

for i, cs in enumerate(cs_por_dia):
    print(f"Dia {i+1}: CS = {cs:.2e} s⁻¹")
    
def diffusion_particle(dp):
    Cc = cunningham(dp)
    return (kB * T * Cc) / (3 * np.pi * mu * dp)

def coag_kernel(dp_i, dp_j):
    Di = diffusion_particle(dp_i)
    Dj = diffusion_particle(dp_j)
    return 4 * np.pi * (Di + Dj) * (dp_i + dp_j)

def calculate_CoagS(df, dp_target_nm=10):
    """
    Coagulation sink para partículas de dp_target_nm
    retorna: Series CoagS(t) em s⁻¹
    """
    dp_bins_nm = df.columns.astype(float).values
    dp_bins_m = dp_bins_nm * 1e-9
    dp_i = dp_target_nm * 1e-9

    CoagS_list = []

    for _, row in df.iterrows():
        Nj = row.values * 1e6  # cm⁻³ → m⁻³
        Kij = coag_kernel(dp_i, dp_bins_m)
        CoagS = np.sum(Kij * Nj)
        CoagS_list.append(CoagS)

    return pd.Series(CoagS_list, index=df.index)

coags_por_dia = [calculate_CoagS(df, dp_target_nm=25).mean()
                 for df in filtered_dfs]

for i, cs in enumerate(coags_por_dia):
    print(f"Dia {i+1}: CoagS(10 nm) = {cs:.2e} s⁻¹")
    
gr_scav_por_dia = [
    calculate_gr_scav(df, coagS.sum(axis=1))
    for df, coagS in zip(filtered_dfs, coagS_por_dia)
]

GR_medio = np.mean(gr_scav_por_dia)
GR_dp = np.std(gr_scav_por_dia, ddof=1)

print(f"GR = {GR_medio:.2f} ± {GR_dp:.2f} nm/h")

cs_por_dia = [calculate_CS(df).mean() for df in filtered_dfs]

CS_medio = np.mean(cs_por_dia)
CS_dp = np.std(cs_por_dia, ddof=1)

print(f"CS = {CS_medio:.2e} ± {CS_dp:.2e} s⁻¹")

CS_medio_10_2 = CS_medio / 1e-2

print(f"CS = {CS_medio_10_2:.2f} × 10^-2 s⁻¹")