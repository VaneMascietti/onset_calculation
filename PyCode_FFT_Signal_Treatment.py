# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:06:43 2025

@author: natal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sympy import symbols, Eq, solve

# REALIZANDO A ATENUAÇÃO DE SINAL POR TRANSFORMADA DE FOURIER #

# Carregar o arquivo CSV (já tirei o cabeçalho, trocar ; por ,) 
df = pd.read_csv(r"250507_EtOH_nC16_0109w.csv", sep=';', header=8, encoding='latin1', on_bad_lines='skip')

# Filtrar PRIMEIRO os dados onde X está entre 20 e 45 (selecionar a rampa não todo experimento)
df_filtrado = df[(df[1].astype(float) >= 20) & (df[1].astype(float) <= 45)].copy()

# Extrair e converter colunas APÓS o filtro 
x = df_filtrado[1].astype(float).values  # SEGUNDA coluna (X)
y_original = df_filtrado[2].astype(float).values  # TERCEIRA coluna (Y original)

# Aplicar a Transformada de Fourier
y_fft = np.fft.fft(y_original)

# Configurar filtro passa-baixo
n = len(y_original) # mostra o número de sinais
freq = np.fft.fftfreq(n) # calcula a frequência para os sinais 
cutoff = 0.005  #FREQUÊNCIA DE CORTE 
mask = np.abs(freq) <= cutoff # máscara booleana para que a frequência seja menor que o corte 

# Aplicar filtro e transformada inversa
y_fft_filtrado = y_fft * mask #retira as frequências acima das desejadas
y_fft_filtrado = np.real(np.fft.ifft(y_fft_filtrado)) #transformada inversa

# Atualizar o DataFrame com os dados filtrados
df_filtrado[2] = y_fft_filtrado

# MÉDIA MOVEL PARA ACHAR PICOS #
# Usar a segunda coluna como X e a terceira como Y
x = df_filtrado.iloc[:, 1]
y = df_filtrado.iloc[:, 2]

# Calcula a média móvel de Y
window_size = 8  # como tem 8198 sinais toma 0,1\%
media_movel = y.rolling(window=window_size, center=True).mean() #cria a janela móvel e atenua

# Salva a média móvel em um novo arquivo CSV
media_movel_df = pd.DataFrame({'X': x, 'Media_Movel': media_movel})
media_movel_df.to_csv('media_movel.csv', index=False)

# Encontra picos na média móvel, tira caso não tenha valor 
media_movel_clean = media_movel.dropna()
peaks, _ = find_peaks(media_movel_clean)

# Ajusta X para coincidir com os índices que tem valor
x_valid = x[media_movel.notna()]

# Encontrar mínimos locais invertendo o sinal da média móvel limpa - vetor
minimos, _ = find_peaks(-media_movel_clean)

# ENCONTRANDO OS PONTOS DE MÍNIMO E MÁXIMO #
# Encontrar o pico que é o máximo global
pico_maximo_global_idx = peaks[np.argmax(media_movel_clean.iloc[peaks])]
x_max_global = x_valid.iloc[pico_maximo_global_idx]
y_max_global = media_movel_clean.iloc[pico_maximo_global_idx]

# Novo DataFrame com X > X do máximo global
df_pos_max_global = media_movel_df[media_movel_df['X'] > x_max_global].dropna()

# Encontrar o segundo pico máximo global nesse novo DataFrame
media_movel_pos = df_pos_max_global['Media_Movel'].reset_index(drop=True)
peaks_pos, _ = find_peaks(media_movel_pos)

# Índice relativo ao novo df, então buscamos o valor original
segundo_max_idx = peaks_pos[np.argmax(media_movel_pos.iloc[peaks_pos])]
x_segundo_max = df_pos_max_global['X'].iloc[segundo_max_idx]
y_segundo_max = df_pos_max_global['Media_Movel'].iloc[segundo_max_idx]

# Novo recorte com X ≤ X do segundo máximo global
df_entre_maximos = media_movel_df[(media_movel_df['X'] >= x_max_global) & (media_movel_df['X'] <= x_segundo_max)].dropna()

# Encontrar o mínimo local com menor valor de Y nesse intervalo
media_movel_entre = df_entre_maximos['Media_Movel'].reset_index(drop=True)
minimos_entre, _ = find_peaks(-media_movel_entre)

# Índice do mínimo com menor valor de Y
minimo_idx = minimos_entre[np.argmin(media_movel_entre.iloc[minimos_entre])]
x_minimo = df_entre_maximos['X'].iloc[minimo_idx]
y_minimo = df_entre_maximos['Media_Movel'].iloc[minimo_idx]

# Resultados
print(f"X do máximo global: {x_max_global}")
print(f"X do segundo máximo global: {x_segundo_max}")
print(f"X do mínimo local entre os dois máximos (menor Y): {x_minimo}")

# REGRESSÃO LINEAR #

df_filtrado[2] = y_fft_filtrado

# Definir variáveis para regressão
x = df_filtrado[1].astype(float).values
y_atenuado = df_filtrado[2].astype(float).values

# Regressão linear entre segundo máximo e mínimo
X1 = np.array([x_segundo_max, x_minimo]).reshape(-1, 1)
y1 = np.array([y_segundo_max, y_minimo])

modelo1 = LinearRegression()
modelo1.fit(X1, y1)

slope1 = modelo1.coef_[0]
intercept1 = modelo1.intercept_
score1 = modelo1.score(X1, y1)

print(f"\nRegressão 1 (segundo máximo até mínimo):")
print(f"Inclinação: {slope1}")
print(f"Intercepto: {intercept1}")
print(f"R²: {score1}")

# Definir intervalo ordenado
x_ini, x_fim = sorted([x_max_global, x_minimo])
mask_central = (x >= x_ini) & (x <= x_fim)
x_sub = x[mask_central]
y_sub = y_atenuado[mask_central]

# 20% centrais
n_total = len(x_sub)
n_central = int(n_total * 0.2)

start = (n_total - n_central) // 2
end = start + n_central

X2 = x_sub[start:end].reshape(-1, 1)
y2 = y_sub[start:end]

modelo2 = LinearRegression()
modelo2.fit(X2, y2)

slope2 = modelo2.coef_[0]
intercept2 = modelo2.intercept_
score2 = modelo2.score(X2, y2)

print(f"\nRegressão 2 (20% centrais):")
print(f"Inclinação: {slope2}")
print(f"Intercepto: {intercept2}")
print(f"R²: {score2}")

# Encontrar ponto de interseção entre as duas retas
x_sym = symbols('x')
eq = Eq(slope1 * x_sym + intercept1, slope2 * x_sym + intercept2)
x_intersec = float(solve(eq, x_sym)[0])
y_intersec = float(slope1 * x_intersec + intercept1)

print(f"\nInterseção das retas:")
print(f"x = {x_intersec:.4f}")
print(f"y = {y_intersec:.4f}")

# RESULTADOS E GRÁFICOS #

# Gráficos
plt.figure(figsize=(18, 5))

# Gráfico 1: Original
plt.subplot(1, 3, 1)
plt.plot(x, y_original, 'b-', label='Original')
plt.title('Dados Originais (X entre 20–45)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(alpha=0.3)
plt.legend()

# Gráfico 2: Filtrado
plt.subplot(1, 3, 2)
plt.plot(x, y_fft_filtrado, 'r-', label='Filtrado (Fourier)')
plt.title('Sinal Filtrado (Passa-Baixo)')
plt.xlabel('X')
plt.ylabel('Y Filtrado')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Plota a média móvel com picos e mínimos locais em um único gráfico
plt.figure(figsize=(12, 6))
plt.plot(x_valid, media_movel_clean, label='Média Móvel', color='blue')
plt.plot(x_valid.iloc[peaks], media_movel_clean.iloc[peaks], 'go', label='Picos (Máximos)')
plt.plot(x_valid.iloc[minimos], media_movel_clean.iloc[minimos], 'ro', label='Mínimos Locais')
plt.title('Média Móvel com Picos e Mínimos Locais')
plt.xlabel('X')
plt.ylabel('Y (Média Móvel)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plota as regressões
plt.figure(figsize=(12, 6))

# Curva suavizada
plt.plot(x, y_atenuado, label='Curva Atenuada (Fourier)', color='blue')

# Pontos importantes
plt.scatter(x_max_global, y_max_global, color='red', label='Máximo Global', zorder=5)
plt.scatter(x_segundo_max, y_segundo_max, color='orange', label='Segundo Máximo', zorder=5)
plt.scatter(x_minimo, y_minimo, color='green', label='Mínimo Local', zorder=5)

# Reta 1 (original no intervalo regressão: segundo máximo -> mínimo)
X1_orig = np.array([x_segundo_max, x_minimo])
y1_orig = slope1 * X1_orig + intercept1
plt.plot(X1_orig, y1_orig, color='red', linestyle='--', label='Regressão 1 (original)')

# Reta 1 extrapolada (do máximo global até segundo máximo)
X1_extra = np.array([x_max_global, x_segundo_max])
y1_extra = slope1 * X1_extra + intercept1
plt.plot(X1_extra, y1_extra, color='red', linestyle='--', label='Regressão 1 (extrapolação)')

# Reta 2 (extrapolada do máximo global até mínimo local)
X2_line = np.array([x_max_global, x_minimo])
y2_line = slope2 * X2_line + intercept2
plt.plot(X2_line, y2_line, color='red', linestyle='--', label='Regressão 2 (extrapolada)')

# Interseção
plt.scatter(x_intersec, y_intersec, color='purple', label='Interseção', s=100, marker='X', zorder=6)

plt.xlabel('X')
plt.ylabel('Y (Curva Atenuada)')
plt.title('Curva Suavizada com Retas de Regressão e Extrapolação')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

