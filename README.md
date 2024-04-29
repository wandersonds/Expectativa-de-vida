
# Expectativa de Vida

Esse Projeto tem como objetivo Prever a Expectativa de Vida Através de Indicadores Socioeconômicos




## Imports

Para rodar esse projeto, você vai precisar Importar as seguintes Bibliotecas

`joblib` `pandas` `numpy` `seaborn` `matplotlib` `sklearn`

```python
# Imports
#sklearn para modelagem ppreditiva 
#stastmodel para modelagem estatistica 

import joblib # Salva o modelo de ML em disco
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor # algoritimo de regressão
from sklearn.preprocessing import StandardScaler  # Padroniza os dados
from sklearn.model_selection import train_test_split # divide em teste e treino
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # algoritmo de regressão
from sklearn.model_selection import GridSearchCV #otimizaçao de hiperparamentros
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
```

## Dicionário de Dados



   - country": "País de origem dos dados - Tipo : object
   - life_expectancy": "Expectativa de vida ao nascer, em anos - Tipo : float64
   - year": "Ano em que os dados foram coletados - Tipo : int64
   - status": "Status de desenvolvimento do país ('Developing' para países em desenvolvimento, 'Developed' para países desenvolvidos) - Tipo : object
   - adult_mortality": "Taxa de mortalidade de adultos entre 15 e 60 anos por 1000 habitantes - Tipo : float64
   - inf_death": "Número de mortes de crianças com menos de 5 anos por 1000 nascidos vivos - Tipo : int64
   - alcohol": "Consumo de álcool per capita (litros de álcool puro por ano) - Tipo : float64
   - hepatitisB": "Cobertura de vacinação contra hepatite B em crianças de 1 ano (%) - Tipo : float64
   - measles": "Número de casos de sarampo relatados por 1000 habitantes - Tipo : int64
   - bmi": "Índice médio de massa corporal da população adulta - Tipo : float64
   - polio": "Cobertura de vacinação contra poliomielite em crianças de 1 ano (%) - Tipo : float64
   - diphtheria": "Cobertura de vacinação contra difteria, tétano e coqueluche (DTP3) em crianças de 1 ano (%) - Tipo : float64
   - hiv": "Prevalência de HIV na população adulta (%) - Tipo : float64
   - gdp": "Produto Interno Bruto per capita (em dólares americanos) - Tipo : float64
   - total_expenditure": "Gasto total em saúde como porcentagem do PIB - Tipo : float64
   - thinness_till19": "Prevalência de magreza em crianças e adolescentes de 10 a 19 anos (%) - Tipo : float64
   - thinness_till9": "Prevalência de magreza em crianças de 5 a 9 anos (%) - Tipo : float64
   - school": "Número médio de anos de escolaridade - Tipo : float64
   - population": "População total do país - Tipo : float64

  ## Distribuição da variável alvo
  ![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/ffe8a438-1c1e-4435-b80e-bcbc24fec482)

  ## Explicando as Funções

  > Função que gera um gráfico para visualizar a relação da variável alvo com as variáveis preditoras e detectar possíveis outliers<BR>
## Relação não gera CASUALIDADE
  ```python
  # Função para o plot da relação da variável alvo com alguns atributos
def get_pairs(data, alvo, atributos, n):
    
    # Grupos de linhas com 3 (n) gráficos por linha
    grupos_linhas = [atributos[i:i+n] for i in range(0, len(atributos), n)]

    # Loop pelos grupos de linhas para criar cada pair plot
    for linha in grupos_linhas:
        plot = sns.pairplot(x_vars = linha, y_vars = alvo, data = data, kind = "reg", height = 3)

    return

    # Verificando outliers
  get_pairs(df_2, alvo, atributos, 3)

  # Aqui temos o relacionamento de seis variáveis com a variável alvo.
  # No eixo Y = Variável alvo
  # No eixo X = Variáveis Preditoras
  ```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/6e8a7f0a-cc95-4f17-a6cf-6624bea40674)


> Função que gera um heatmap para visualizar a relação da variável alvo com as variáveis preditoras superiores ou inferiores a 0.3 (positivas ou negativas).<BR>

  ```python
  # Função para filtrar e visualizar correlação
def filtrar_e_visualizar_correlacao(df, threshold, drop_column = None):

    # Calcula a matriz de correlação
    corr = df.corr()
    
    # Aplica os filtros de limiar, excluindo a correlação perfeita
    filtro = (abs(corr) >= threshold) & (corr != 1.0)
    df_filtrado = corr.where(filtro).dropna(how = 'all').dropna(axis = 1, how = 'all')
    
    # Remove a coluna e linha especificada, se fornecido
    if drop_column:
        df_filtrado = df_filtrado.drop(index = drop_column, 
                                       errors = 'ignore').drop(columns = drop_column, 
                                                               errors = 'ignore')
    
    # Visualiza o resultado com um heatmap somente com as variáveis que satisfazem o critério de filtro
    plt.figure(figsize = (8, 6))
    sns.heatmap(df_filtrado, annot = True, cmap = 'coolwarm', center = 0)
    plt.show()

  # Executa a função
  filtrar_e_visualizar_correlacao(df_3, threshold = 0.3, drop_column = None) # se existir alguma correlacao 
                                                                                      # maior ou menor que 03 mostra no grafico
    
  # correlacao com as variaveis preditoras com a variavel alvo
  ```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/f8392a63-2661-41b8-88b0-e8246e60d596)

> Modificando o filtro da função para excluir a variável alvo e verificar a alta correlação entre as variáveis preditoras, a fim de prevenir a multicolinearidade. Vamos considerar 0.55 como o ponto de corte.
```python
  # Executa a função
  # correlacao com as variaveis preditoras com as variaveis preditoras
  filtrar_e_visualizar_correlacao(df_3, threshold = 0.55, drop_column = 'life_expectancy')
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/ef5e5090-ca84-4b29-b7d0-87681ca736db)
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/f1c8a7de-a9d3-46d1-b678-aecc43f70609)
```python
# Preparando o novo dataset
df_final = pd.DataFrame({'life_expectancy': df_3['life_expectancy'],
                              'adult_mortality': df_3['adult_mortality'],
                              'diphtheria': df_3['diphtheria'],
                              'hiv': df_3['hiv'],
                              'gdp': df_3['gdp'],
                              'thinness_till19': df_3['thinness_till19'],
                              'school': df_3['school'],
                              'lifestyle': df_3['lifestyle'],})
df_final.head()
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/cf43d4e7-8c1f-4714-9eb9-e317b6f5ba33)

## Seleção do Modelo

```python
#AQUI EU QUERO O MENOR ERRO POSSIVEL
print('RMSE V1:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v1)))
print('RMSE V2:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v2)))
print('RMSE V3:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v3)))
print('RMSE V4:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v4)))
print('RMSE V5:', np.sqrt(metrics.mean_squared_error(y_teste, y_pred_teste_v5)))
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/a6508826-0827-445c-abf7-928245697eeb)

```python
#QUANTO MAIOR MELHOR
print('R2 Score Modelo V1:', metrics.r2_score(y_teste, y_pred_teste_v1))
print('R2 Score Modelo V2:', metrics.r2_score(y_teste, y_pred_teste_v2))
print('R2 Score Modelo V3:', metrics.r2_score(y_teste, y_pred_teste_v3))
print('R2 Score Modelo V4:', metrics.r2_score(y_teste, y_pred_teste_v4))
print('R2 Score Modelo V5:', metrics.r2_score(y_teste, y_pred_teste_v5))
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/b881c488-91ad-4e89-809e-4849d181bdb6)

## Deploy e Uso do Modelo Para Previsão com Novos Dados
```python
# Carrega padronizador e modelo
scaler_final = joblib.load('scaler.pkl')
modelo_final = joblib.load('modelo_v1.pkl')
# Carregando os novos dados
novos_dados = pd.read_csv('novos_dados.csv')
# Visualiza
novos_dados
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/7b117f49-ade4-4492-943d-af12f74cbc72)

```python
# Os novos dados precisam ser padronizados
novos_dados_scaled = scaler_final.transform(novos_dados)
# Visualiza
novos_dados_scaled
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/2f86b2b1-edcc-41f1-964c-1e7151a2b9ed)

```python
# Previsão
previsao = modelo_final.predict(novos_dados_scaled)
type(previsao)
```
```python
print('De acordo com os dados de entrada a expectativa de vida (em anos) é de aproximadamente:', 
      np.round(previsao, 2)))
```
![image](https://github.com/wandersonds/Expectativa-de-vida/assets/143017669/efba1166-d727-4ce8-9872-f86c71413883)





  




