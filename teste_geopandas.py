import geopandas as gpd
import matplotlib.pyplot as plt

# Carrega o arquivo shapefile com as geometrias dos municípios brasileiros
br_municipios = gpd.read_file("C:/Users/allan/Desktop/dissertacao/shape_bacia_do_parnaiba/bacia_parnaiba.shp")

# Cria um novo DataFrame com dados fictícios de população para cada município
populacao = gpd.GeoDataFrame({
    'codigo_ibge': [3106200, 3106309, 3106408, 3106507],
    'nome': ['Belo Horizonte', 'Contagem', 'Betim', 'Juiz de Fora'],
    'populacao': [2512071, 668089, 429475, 571128],
    'geometry': [
        br_municipios.loc[br_municipios['NM_MUNICIP'] == 'Belo Horizonte', 'geometry'].values[0],
        br_municipios.loc[br_municipios['NM_MUNICIP'] == 'Contagem', 'geometry'].values[0],
        br_municipios.loc[br_municipios['NM_MUNICIP'] == 'Betim', 'geometry'].values[0],
        br_municipios.loc[br_municipios['NM_MUNICIP'] == 'Juiz de Fora', 'geometry'].values[0]
    ]
})

# Plota o mapa
fig, ax = plt.subplots(figsize=(10, 10))

# Configura a cor dos municípios de acordo com a população
populacao.plot(column='populacao', cmap='Blues', linewidth=0.8, edgecolor='gray', ax=ax)

# Configura o título e a legenda do mapa
ax.set_title('População dos municípios de Minas Gerais', fontsize=16)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.axis('off')

# Mostra o mapa
plt.show()
