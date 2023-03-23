import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Carregue um arquivo CSV com as informações das coordenadas e do valor
df = pd.read_csv("C:/Users/allan/Documents/GitHub/Dissertacao_mestrado_allan/codigo_dados_imagens/dados/grid_processado.txt", sep = '\t')

colorscale = [[0, 'rgb(255, 255, 204)'], 
              [0.2, 'rgb(255, 237, 160)'], 
              [0.4, 'rgb(254, 217, 118)'], 
              [0.6, 'rgb(254, 178, 76)'], 
              [0.8, 'rgb(253, 141, 60)'], 
              [1, 'rgb(252, 78, 42)']]


fig = go.Figure(data=go.Scattergeo(
        lat=df['Lat'],
        lon=df['Lon'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['h'],
            colorscale=colorscale,
            line_width=0
        )
    ))

fig.update_layout(
        geo=dict(
            scope='world',
            projection_type='equirectangular',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            subunitcolor='rgb(255, 255, 255)',
            countrycolor='rgb(255, 255, 255)',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showsubunits=True,
            showcountries=True,
            resolution=50,
            projection_rotation=dict(lon=15, lat=10),
            coastlinewidth=1,
            coastlinecolor='rgb(255, 255, 255)',
            bgcolor='rgba(0,0,0,0)'
        ),
        title='Mapa temático de gravidade',
    )


fig.show()