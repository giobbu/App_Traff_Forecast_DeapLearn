import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import pydeck as pdk
import yaml


def initial_layer_deck():
        INITIAL_VIEW_STATE = pdk.ViewState(latitude=50.85045, longitude=4.34878, zoom=8, max_zoom=8, pitch=45, bearing=0)
        r = pdk.Deck(initial_view_state=INITIAL_VIEW_STATE, map_style='mapbox://styles/mapbox/light-v9')
        return r

def update_layer_deck(lst, streets, pred):

        STREETS = [int(float(s)) for s in lst]
        df = streets[streets.index.isin(STREETS)]
        df['flow'] =  pd.DataFrame(pred).loc[0].astype(float).values

        INITIAL_VIEW_STATE = pdk.ViewState(latitude=50.85045, longitude=4.34878, zoom=8, max_zoom=8, pitch=45, bearing=0)

        geojson = pdk.Layer(
                "GeoJsonLayer",
                df,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation = "flow*30",
                get_fill_color='[255, (1-flow/250)*255, 0]',
                get_line_color='[255, 255, 255]',
                pickable=True
                )

        r = pdk.Deck(layers=[geojson], 
                         initial_view_state=INITIAL_VIEW_STATE,
                         map_style='mapbox://styles/mapbox/light-v9',
                         tooltip={"text": "{flow}"})

        return r

with open('config.yaml') as file:
        config = yaml.safe_load(file)

path = config['script_path']

