
import folium
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import folium_static
import pydeck as pdk

def plot_network(file):
        df_belgium = gpd.read_file(file)
        m = folium.Map([50.85045, 4.34878], zoom_start=9, tiles='cartodbpositron')
        folium.GeoJson(df_belgium).add_to(m)
        return m, df_belgium


def plot_loss(tr_loss, val_loss):
        fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
        axes.set_ylabel("Loss (MAE)", fontsize=14)
        axes.plot(tr_loss)
        axes.plot(val_loss)
        axes.set_xlabel("Epoch", fontsize=14)
        leg = axes.legend(loc='upper right')
        return st.pyplot(fig)


def plot_deck(streets):

        INITIAL_VIEW_STATE = pdk.ViewState(latitude=50.85045, longitude=4.34878, zoom=9, max_zoom=9, pitch=45, bearing=0)
        geojson = pdk.Layer(
        "GeoJsonLayer",
        streets['geometry'],
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation= 10,
        get_line_color=[255, 255, 255],
        get_fill_color=[255, 255, 255]
        )

        r = pdk.Deck(layers=geojson, initial_view_state=INITIAL_VIEW_STATE, map_style='mapbox://styles/mapbox/light-v9')
        return r