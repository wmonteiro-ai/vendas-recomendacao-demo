import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

##########
#Streamlit setup
##########
st.set_page_config(page_title='Inteledge - Recomendação de Vendas', page_icon="🛒", layout="centered", initial_sidebar_state="auto", menu_items=None)

##########
#Functions for the predictions and for the page layout
##########
@st.cache(allow_output_mutation=True)
def get_pickles():
	df_products_metadata = pickle.load(open('data.pkl', 'rb'))
	index = tf.saved_model.load('sequential_model.tf')
		
	return df_products_metadata[0].sort_values(by='movie_title'), index

def get_predictions(df_products_metadata, index, product_order):
	items_to_evaluate = product_order + [b'0'] * 10
	items_to_evaluate = tf.concat([[items_to_evaluate[:10]]], axis=1)

	ratings, items = index(items_to_evaluate)
	return df_products_metadata.loc[[x.decode() for x in product_order if x!=b'0']], \
			df_products_metadata.loc[[x.decode() for x in items.numpy()[0] if x!=b'0']]

##########
#Preparing the simulator
##########
# loading the predictive model and the underlying dataset
df_products_metadata, index = get_pickles()

##########
#Section 1 - Simulator
##########
col1, _, _ = st.columns(3)
with col1:
	st.image('inteledge.png')

st.title('Inteligência Artificial de Recomendação')
st.markdown('Este é um exemplo que fizemos de um sistema de recomendação de produtos. Neste caso, utilizamos para esta demonstração um histórico de filmes assistidos e avaliados por diversos usuários ao longo do tempo.')
st.markdown('É legal você saber que já criamos no passado algoritmos deste mesmo tipo usando, como base, o histórico de clientes das áreas da indústria e do varejo; de clientes que não deixam notas (mas que apenas compram/deixam de comprar produtos sem feedback algum); ou, ainda, sugestão de produtos recomendados para novos clientes. Estes outros algoritmos que criamos usam também dados socioeconômicos e demais informações cadastrais dos clientes.')
st.markdown('Este tipo de inteligência também pode ser usada em aplicações como e-commerce e em sistemas educacionais. Ficou interessado? Siga-nos em @inteledge.app no [Instagram](https://instagram.com/inteledge.app) e no [LinkedIn](https://www.linkedin.com/company/inteledge/)!')

st.subheader('Histórico de filmes')
st.markdown('Aqui, filtramos somente pelos filmes lançados entre 1900 a 2000. Selecione os filmes que assistiu, com os que viu (e *gostou*) mais recentemente em primeiro lugar. Consideramos os 10 primeiros filmes, mas você pode informar menos se quiser.')
st.markdown('Também fazemos isso em sistemas de e-commerce (com algoritmos do tipo *"Clientes que compraram isso também compram estes outros produtos"* ou *"Você provavelmente também irá querer estes produtos"*). Procuramos sempre maximizar o ticket médio dos seus clientes e também que sintam que os resultados estejam sendo efetivos para eles.')
st.markdown('❗ ***Lembre-se:*** quanto mais próximo de 10 filmes selecionados, melhores serão as predições.')
products = dict(zip(df_products_metadata.index, df_products_metadata['movie_title'].values))
chosen_products = st.multiselect('Informe aqui os seus filmes.', products.values())
product_order = df_products_metadata[df_products_metadata['movie_title'].isin(chosen_products)].index.tolist()
product_order = [s.encode('utf-8') for s in product_order]

# variables
st.write()

# inference
df_history, df_recommendations = get_predictions(df_products_metadata, index, product_order)

# history
st.subheader('Recomendações de novos filmes')
st.markdown('Estes são os próximos filmes que você **provavelmente** assistiria a partir da lista informada acima.')
st.table(df_recommendations.reset_index(drop=True).rename(columns={'movie_title': 'Você provavelmente gostará de assistir:'}))

st.markdown('Veja que as recomendações atualizam *instantaneamente* conforme você adiciona ou remove filmes. Imagine que isto se aplica também a supermercados (para sugerir mais produtos aos consumidores que provavelmente estavam se esquecendo), lojas de departamento (para sugerir mais produtos, atacado (para auxiliar na venda para outras empresas), entre outros. Este é só um tipo de algoritmo de recomendação que existe, mas aqui na **inteledge.** temos experiência em vários outros tipos. Customizamos e preparamos algoritmos especialmente para você e para os seus casos de uso.')

st.image('photo.jpg', 'Photo by NeONBRAND on Unsplash')

st.markdown('Siga-nos no [Instagram](https://instagram.com/inteledge.app) e no [LinkedIn](https://www.linkedin.com/company/inteledge/)!')
