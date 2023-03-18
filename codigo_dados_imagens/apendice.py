-----------------------------------------------------------------
# @Autor: Allan Soares Ramalho
-----------------------------------------------------------------
# Importando as bibliotecas necessarias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import datasets
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from functions.ellipsoid import WGS84
from functions.grav_func import gamma_closedform
-----------------------------------------------------------------
# Lendo arquivo do funcional gravity earth e conversao das longitudes

header=['Lon', 'Latitude', 'Orthom', 'Gabs']
gravity_earth=pd.read_csv('dados/gravity_earth.gdf',
                          sep='\s+',
                          skiprows=34,
                          names=header)
Lon=np.array(gravity_earth['Lon'])
Longitude=np.zeros(len(gravity_earth))
for i in range (len(gravity_earth)):
    Longitude[i] = Lon[i]-360
gravity_earth['Longitude'] = Longitude
-----------------------------------------------------------------
# Lendo os valores de N calculados no MAPGEO2015 e calculo de h da malha

header2=['Geoid_und']
ondulacao=pd.read_csv('dados/ondulacao_geoidal_grid.txt',
                      sep='\s+',
                      decimal=b'.',
                      names=header2)
N=np.array(ondulacao['Geoid_und'])
gravity_earth['Geoid_und']=N
gravity_earth['Geom'] = gravity_earth['Orthom'] + gravity_earth['Geoid_und']
-----------------------------------------------------------------
# Calculo da gravidade normal, a partir da funcao 'grav_func.py'

g_norm=gamma_closedform(gravity_earth['Geom'],gravity_earth['Latitude'])
gravity_earth['Gnorm'] = g_norm
-----------------------------------------------------------------
# Calculo do disturbio de gravidade da malha regular

delta_grid = gravity_earth['Gabs'] - gravity_earth['Gnorm']
gravity_earth['Disturb'] = delta_grid
-----------------------------------------------------------------
# lendo o arquivo dos dados observados

terrestre=pd.read_csv('dados/Parnaiba_grav_survey.txt',
                      sep=',', 
                      usecols=(0,7,16,17,19))
-----------------------------------------------------------------
# Calculo da gravidade normal observada

g_norm_c=gamma_closedform(terrestre['Elevation'], terrestre['Latitude'])
terrestre['g_norm'] = g_norm_c
-----------------------------------------------------------------
# Calculo dos disturbios observados

delta_obs_c = terrestre['Gravity'] - terrestre['g_norm']
terrestre['delta_obs']=delta_obs_c
-----------------------------------------------------------------
# Lendo os dados preditos pelo funcional gravity

header=['Lon', 'Lat', 'h', 'g_abs_pred']
terrestre_sat=pd.read_csv('dados/output_icgem_caminhamento_predito.dat',
                          skiprows=32, 
                          sep='\s+',
                          decimal=b'.',
                          names=header, 
                          usecols=(1,2,3,4))
-----------------------------------------------------------------
# Calculo gravidade normal predita 

g_norm_predito=gamma_closedform(terrestre_sat['h'], terrestre_sat['Lat'])
terrestre_sat['g_norm_pred'] = g_norm_predito
-----------------------------------------------------------------
# Calculo do disturbio predito 

delta_pred_c = terrestre_sat['g_abs_pred'] - terrestre_sat['g_norm_pred']
terrestre_sat['delta_pred']=delta_pred_c
-----------------------------------------------------------------
# Unificando em um unico data frame os dados preditos e dados observados para o caminhamento

terrestre['g_abs_pred']=terrestre_sat['g_abs_pred']
terrestre['g_norm_pred']=terrestre_sat['g_norm_pred']
terrestre['delta_pred']=terrestre_sat['delta_pred']
-----------------------------------------------------------------
# Calculo dos Residuos de disturbios de gravidade

R=terrestre['delta_obs'] - terrestre['delta_pred']
terrestre['residuos']=R
-----------------------------------------------------------------
# Imagem da Bacia do Parnaiba e estacoes de levantamento

fig = plt.figure(figsize=(14,14))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-49.5, -39.5, -1.5, -11.78], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=1)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='black',
                  alpha=0.5)

# Adicionando o scatter com as informacoes dos dataframes
ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
             color='red',
             transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}
plt.savefig('imagens/bacia do parnaiba.png',format= 'png',dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de h predita para a malha

fig = plt.figure(figsize=(14,14))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50.4,-39,-12.2,0.2], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.8)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_earth['Geom'],
              cmap='terrain',
              vmin=-27, vmax=1309,
              transform=ccrs.PlateCarree())


# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-27, 1309, 7, endpoint=True)
cbar=plt.colorbar(f2,shrink=0.7,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar.set_label('Altitude geometrica (m)',fontsize=20,labelpad=2)
plt.savefig('imagens/altitude geometrica da bacia do parnaiba.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de Gravidade absoluta

fig = plt.figure(figsize=(14,14))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50.4,-39,-12.2,0.2], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linestyle='--',
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_earth['Gabs'],
              cmap='RdBu_r',
              vmin=977811, vmax=978185,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(977811, 978185, 5, endpoint=True)
cbar=plt.colorbar(f2,shrink=0.7,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.set_label('Gravidade absoluta (mGal)',fontsize=20,labelpad=2)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
plt.savefig('imagens/gravidade absoluta da bacia do parnaiba.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de Gravidade normal

fig = plt.figure(figsize=(14,14))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50.4,-39,-12.2,0.2], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True, 
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_earth['Gnorm'],
              cmap='RdBu_r',
              vmin=977774, vmax=978196,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(977774, 978196, 5, endpoint=True)
cbar=plt.colorbar(f2,shrink=0.7,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.set_label('Gravidade normal (mGal)',fontsize=20,labelpad=2)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
plt.savefig('imagens/gravidade normal da bacia do parnaiba.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de Disturbio

fig = plt.figure(figsize=(14,14))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50.4,-39,-12.2,0.2], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True, 
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)
                  
# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_e\rth['Disturb'],
              cmap='RdBu_r',
              vmin=-100, vmax=100,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-100, 100, 7, endpoint=True)
cbar.set_label('$\delta_{g}$ (mGal)',fontsize=20,labelpad=2)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar=plt.colorbar(f2,shrink=0.7,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.set_label('Disturbio de gravidade (mGal)',fontsize=20,labelpad=2)
plt.savefig('imagens/disturbio da bacia do parnaiba.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de h observado

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True,
                  linestyle='--',
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
              c=terrestre['Elevation'], 
              s=100,
              cmap='terrain',
              vmin=48 ,vmax=719,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(48, 719, 7, endpoint=True)
cbar=plt.colorbar(f3,shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar.set_label('Altitude geometrica (m)',fontsize=20,labelpad=2)
plt.savefig('imagens/altitude geom campo.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de h predito e h observado

fig = plt.figure(figsize=(14,12))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_earth['Geom'],
              cmap='terrain',
              vmin=-27,vmax=1309,
              alpha=0.1,
              transform=ccrs.PlateCarree())

f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
              c=terrestre['Elevation'],
              s=100,
              cmap='terrain',
              vmin=-27,vmax=1309,
              alpha=1,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-27, 1309, 7, endpoint=True)
cbar=plt.colorbar(f3,shrink=0.85,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.set_label('Altitude geometrica (m)',fontsize=20,labelpad=2)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
plt.savefig('imagens/altitude grid x campo.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa dos disturbios preditos

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  linestyle='--',
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
              c=terrestre['delta_pred'],
              s=100, 
              cmap='RdBu_r',
              vmin=-57,vmax=57,
              transform=ccrs.PlateCarree())
              
# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False  
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-57, 57, 7, endpoint=True)
cbar=plt.colorbar(f3, shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar.set_label('Disturbio de gravidade (mGal)',fontsize=20,labelpad=2)
plt.savefig('imagens/perfil disturbio predito.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa dos disturbios observados

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True, 
                  linestyle='--',
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'], c=terrestre['delta_obs'], 
              s=100, 
              cmap='RdBu_r', 
              vmin=-57,vmax=57,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-57, 57, 7, endpoint=True)
cbar=plt.colorbar(f3, shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar.set_label('Disturbio de gravidade (mGal)',fontsize=20,labelpad=2)
plt.savefig('imagens/perfil disturbio observado.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa dos residuos de disturbio de gravidade

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 

ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
              c=terrestre['residuos'],
              s=100,
              vmin=-22.5,vmax=22.5,
              cmap='RdBu_r', transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-22.5, 22.5, 5, endpoint=True)
cbar=plt.colorbar(f3, shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
cbar.set_label('Residuos (mGal)',fontsize=20,labelpad=5)
plt.savefig('imagens/perfil residuos.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de disiturbios observados e preditos

fig = plt.figure(figsize=(14,12))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.add_feature(cfeature.BORDERS)
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

#Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black', 
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f2=ax.scatter(gravity_earth['Longitude'], gravity_earth['Latitude'],
              c=gravity_earth['Disturb'],
              cmap='RdBu_r',
              vmin=-70,vmax=70,
              alpha=0.07,
              transform=ccrs.PlateCarree())

f3=ax.scatter(terrestre['Longitude'], terrestre['Latitude'],
              s=200,
              c=terrestre['delta_obs'], 
              cmap='RdBu_r', 
              vmin=-70,vmax=70,
              alpha=1.0,
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-70 ,70, 7, endpoint=True)
cbar=plt.colorbar(f3,shrink=0.85,orientation='horizontal',pad=0.07,aspect=30, ticks=v)
cbar.set_label('Disturbio de gravidade (mGal)',fontsize=20,labelpad=2)
cbar.ax.tick_params(labelsize=14,color='black',labelcolor='black')
plt.savefig('imagens/disturbio obs X pred.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Lendo arquivo de h predito do funcional h_topo_over_ell

header=['Longitude','Latitude','Geom']
h_predito = pd.read_csv('dados/output_icgem_altitude_geometrica_estacoes.dat',skiprows=38, sep='\s+',decimal=b'.',names=header, usecols=(1,2,4))
terrestre['h_pred']=h_predito['Geom']
-----------------------------------------------------------------
# Calculo do residuo de h

terrestre['residuos_h'] = terrestre.Elevation - terrestre.h_pred
-----------------------------------------------------------------
# Mapa de h predito para o caminhamento

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre.Longitude, terrestre.Latitude,
              c=terrestre.h_pred, 
              s=100,
              vmin=48,vmax=719,
              cmap='terrain', 
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima
g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(48, 719, 7, endpoint=True)
cbar=plt.colorbar(f3, shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.set_label('Altitude geometrica (m)',fontsize=22,labelpad=2)
cbar.ax.tick_params(labelsize=12,color='black',labelcolor='black')
plt.savefig('imagens/altitude geom predita campo.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Mapa de residuos de h

fig = plt.figure(figsize=(16,16))

# Importando o shape da bacia do parnaiba
fname='shape_bacia_do_parnaiba/bacia_parnaiba.shp' 
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Feicoes
ax.set_extent([-50, -39.5, -3.5, -7.5], ccrs.PlateCarree())

# Determinando limite dos estados
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_shp',
                                      scale='50m',
                                      facecolor='none')

# Adicionando a feicao dos estados
ax.add_feature(states, edgecolor='gray',linestyle=':', linewidth=2,alpha=0.4)

# Adicionando a geometria do shape da bacia do parnaiba
ax.add_geometries(Reader(fname).geometries(),
                  ccrs.PlateCarree(),
                  facecolor='none',
                  edgecolor='black',
                  linewidth=4,
                  alpha=0.5)

# Adicionando linhas de grade
g1 = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True,
                  linestyle='--', 
                  linewidth=1,
                  color='gray',
                  alpha=0.8)

# Adicionando o scatter com as informacoes dos dataframes
f3=ax.scatter(terrestre.Longitude, terrestre.Latitude,
              c=terrestre.residuos_h,
              s=100,
              vmin=-62.5,vmax=62.5,
              cmap='terrain',
              transform=ccrs.PlateCarree())

# Removendo os eixos do lado direito e de cima

g1.ylabels_right = False
g1.xlabels_top = False

# Formatando os eixos para georreferenciar
g1.yformatter = LATITUDE_FORMATTER
g1.xformatter = LONGITUDE_FORMATTER
g1.xlabel_style = {'size': 15}
g1.ylabel_style = {'size': 15}

v = np.linspace(-62.5, 62.5, 7, endpoint=True)
cbar=plt.colorbar(f3, shrink=0.85,orientation='horizontal',pad=0.07,aspect=30,ticks=v)
cbar.ax.tick_params(labelsize=12,color='black',labelcolor='black')
cbar.set_label('Residuos (m)',fontsize=18, labelpad=2)
plt.savefig('imagens/residuos altitude geom.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Hitograma de residuos de disturbio de gravidade

ndat = len(terrestre.residuos)
mean = terrestre['residuos'].mean()
std = terrestre['residuos'].std()
vmin = min(terrestre.residuos)
vmax = max(terrestre.residuos)
text_mean = '%1.2f' % mean
text_std = '%1.2f' % std

fig = plt.figure(figsize=(14,12))
plt.style.use('ggplot')
plt.hist(terrestre.residuos, bins=50, density=True,color='#0504aa', alpha=0.77, rwidth=0.85)
plt.xlim(-8,23)
plt.ylim(0,0.12)
x = np.linspace(vmin,vmax, ndat,endpoint=True)
y = norm.pdf(x, np.mean(terrestre.residuos), np.std(terrestre.residuos))
plt.text(-5,0.09,'media='+text_mean,fontsize=20)
plt.text(-5,0.10,'desvio padrao='+text_std,fontsize=20)
plt.xlabel('Residuos (mGal)' , fontsize = 22, labelpad = 16)
plt.ylabel('Frequencia (%)', fontsize = 22, labelpad = 16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.plot(x, y, '--r', linewidth=2)
plt.savefig('imagens/histograma_residuos_delta.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Hitograma de residuos de h

ndat = len(terrestre.residuos_h)
mean = terrestre['residuos_h'].mean()
std = terrestre['residuos_h'].std()
vmin = min(terrestre.residuos_h)
vmax = max(terrestre.residuos_h)
text_mean = '%1.2f' % mean
text_std = '%1.2f' % std
fig = plt.figure(figsize=(14,12))
plt.style.use('ggplot')
plt.hist(terrestre.residuos_h, bins=100, density=True,color='green', alpha=0.77, rwidth=0.85)
plt.xlim(-30,60)
plt.ylim(0,0.053)
x = np.linspace(vmin,vmax, ndat,endpoint=True)
y = norm.pdf(x, np.mean(terrestre.residuos_h), np.std(terrestre.residuos_h))
plt.text(-20,0.035,'media='+text_mean,fontsize=20)
plt.text(-20,0.04,'desvio padrao='+text_std,fontsize=20)
plt.xlabel('Residuos (m)' , fontsize = 22, labelpad = 16)
plt.ylabel('Frequencia (%)', fontsize = 22, labelpad = 16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.plot(x, y, '--r', linewidth=2)
#plt.title('Histograma dos residuos ', fontsize=20, y=1.02)
plt.savefig('imagens/histograma_residuos_h.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Grafico de dispersao delta obs x delta pred

plt.figure(figsize=(14,12))

xd = terrestre.delta_obs
yd = terrestre.delta_pred
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='blue')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)

plt.text(20,0,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.xlabel("$\delta g_{\,\,observado}$ (mGal)", fontsize=22,labelpad=16)
plt.ylabel("$\delta g_{\,\,predito}$ (mGal)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao delta obs x pred.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# grafico de dispersao h observado x h predito

plt.figure(figsize=(14,12))

xd = terrestre.Elevation
yd = terrestre.h_pred
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='green')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)
plt.text(600,520,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.xlabel("$h_{\,observada}$ (m)", fontsize=22,labelpad=16)
plt.ylabel("$h_{\,predita}$ (m)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao h obs x pred.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Grafico de dispersao delta obs x h obs

plt.figure(figsize=(14,12))

yd = terrestre.Elevation
xd = terrestre.delta_obs
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='blue')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)
plt.text(20,420,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.xlabel("$\delta g_{\,\,observado}$ (mGal)", fontsize=22,labelpad=16)
plt.ylabel("$h_{\,\,observada}$ (m)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.xlim(-57,40)
plt.ylim(0,720)
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao delta obs x h obs.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Grafico de dispersao delta pred x h pred

plt.figure(figsize=(14,12))

yd = terrestre.h_pred
xd = terrestre.delta_pred
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='green')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)
plt.text(12,420,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.xlabel("$\delta g_{\,\,predito}$ (mGal)", fontsize=22,labelpad=16)
plt.ylabel("$h_{\,\,predita}$ (m)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
lt.xlim(-57,40)
plt.ylim(0,720)
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao delta pred x h pred.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
# Grafico de dispersao delta pred x h obs

plt.figure(figsize=(14,12))

yd = terrestre.Elevation
xd = terrestre.delta_pred
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='purple')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)
plt.text(10,420,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.xlabel("$\delta g_{\,\,predito}$ (mGal)", fontsize=22,labelpad=16)
plt.ylabel("$h_{\,\,observada}$ (m)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.xlim(-57,40)
plt.ylim(0,720)
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao delta pred x h obs.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------
#grafico de dispersao residuos h e residuos de disturbio

plt.figure(figsize=(14,12))

data = pd.DataFrame(terrestre, columns=['Elevation','h_pred'])
xd = terrestre.residuos
yd = terrestre.residuos_h
reorder = sorted(range(len(xd)), key = lambda ii: xd[ii])
xd = [xd[ii] for ii in reorder]
yd = [yd[ii] for ii in reorder]
plt.scatter(xd, yd, alpha=0.4, marker='o', color='black')

# Determinar a melhor linha de ajuste
par = np.polyfit(xd, yd, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]

# Coeficiente de determinacao
variance = np.var(yd)
residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
Rsqr = np.round(1-residuals/variance, decimals=2)
plt.text(19,40,'$R^2 = %0.2f$'% Rsqr, fontsize=20)
plt.ylabel("residuos h (m)", fontsize=22,labelpad=16)
plt.xlabel("residuos $\delta g$ (mGal)",fontsize=22,labelpad=16)
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.plot(xl, yl, '-r')
plt.savefig('imagens/dispersao residuos.png',format='png', dpi=300, bbox_inches='tight')
plt.show()
-----------------------------------------------------------------