# importando as bibliotecas necessarias

import numpy as np
import pandas as pd
from functions.ellipsoid import WGS84
from functions.grav_func import gamma_closedform

def residuo(dft,dfs):
    '''Função que faz o cálculo de resíduos de gravidade.
    
    input>
    
    dft:   dataframe  -> dataframe do arquivo carregado com os dados observados em campo
    dfs:   dataframe  -> dataframe do arquivo carregado com os dados preditos pelo modelo
   '''
    
    #calculando a gravidade normal observada e incorporando ao DataFrame
    
    g_norm_t=gamma_closedform(dft['Elevation'], dft['Latitude'])
    dft['Normal_gravity']=g_norm_t
    #------------------------------------------------------------------------------
    
    # Calculo do disturbio observado e incorporando ao dataframe
    
    delta_obs_t = dft.Gravity - dft.Normal_gravity
    dft['Delta']=delta_obs_t
    #-------------------------------------------------------------------------------
    
    
    #calculando a gravidade normal predita e incorporando ao DataFrame
    
    g_norm_s=gamma_closedform(dfs['Elevation'], dfs['Latitude'])
    dfs['Normal_gravity']=g_norm_s
    #------------------------------------------------------------------------------
    
    # Calculo do disturbio observado e incorporando ao dataframe
    
    delta_pred = dfs.Gravity - dfs.Normal_gravity
    dfs['Delta']=delta_pred
    #-------------------------------------------------------------------------------
    
    # Calculo dos Residuos = Disturbio observado - disturbio predito e incorporando aos dataframes
    R=dft.Delta - dfs.Delta
    dft['Residual']=R
    dfs['Residual']=R