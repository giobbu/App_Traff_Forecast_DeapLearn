
# ML packages
import yaml
from util import logging, load_streets
from app_util import plot_network, plot_loss, plot_deck

from data.data_module import data_reader, feat_engin, data_split_scale, data_loader
from data.app_util import plot_distribution, plot_pie
from model.model_module import LSTM_ED
from training.train_module import training
from training.util import opt, loss_fct, valid_loss_fct
from training.app_util import slider_display, save_updates_yaml, upload_yaml
from testing.test_module import testing


import time
import os
import numpy as np
import tensorflow as tf
import folium
import geopandas as gpd

# App packages
import streamlit as st
import SessionState
import pydeck as pdk
import matplotlib.pyplot as plt

session_state = SessionState.get(check1=False)

file_streets = 'data/Belgium_streets.json'
checkpoint_dir = 'trained_model_dir'


with open('config.yaml') as file:
        config = yaml.safe_load(file)


path = config['script_path']
mean_value = config['data']['threshold']
n_feat_time = config['data']['time_feature']
validation_period = config['data']['validation']
testing_period = config['data']['testing']

inp_sqc = config['loader']['input_sqc']
out_sqc = config['loader']['output_sqc']
total_dim = config['loader']['tot_dim']
aux_dim = n_feat_time
batch_tr = config['loader']['batch_tr']
batch_vl = config['loader']['batch_tr']
batch_ts =  config['loader']['batch_ts']






def main():

        st.set_page_config(page_title="APP")

        st.markdown("""---""")
        # Space out the maps so the first one is 2x the size of the other three
        with st.beta_container():

                st.title('LSTM-Encoder Decoder for Freight Traffic Forecasting')
                st.write('Traffic flow forecasting is fundamental to todays Intelligent Transportation Systems (ITS). In this context deep learning models have recently shown promising results. The study presents a LSTM encoder-decoder for multi-horizon traffic flow predictions.')
                st.write("Check the paper [HERE](https://www.researchgate.net/publication/348930068_A_Tutorial_on_Network-Wide_Multi-Horizon_Traffic_Forecasting_with_Deep_Learning)")
 
                st.markdown("""---""")

                # # # visualize streets
                # st.header('Belgian Freeway System Map')
                # map_txt = st.text('Loading map with streets...') 
                # streets = load_streets(file_streets)
                # m = folium.Map([38.8934, -76.9470],  zoom_start=12)
                # st.markdown(m.repr_html(), unsafe_allow_html = True)
                # st.markdown("""---""")

        
       
        # map = st.empty()
        # check = st.empty()
        # for i in range(2):
        #         check.text('Loading map with streets...') 
        #         r = plot_deck(streets)
        #         map.pydeck_chart(r)
        #         r.update()
        #         check.text(i)
        #         time.sleep(10)
        # map_txt.text('Loading map with streets...done!')


                col0, col1, col2 = st.beta_columns(3)

                with col0:
                        # read OBU data file
                        st.subheader('Raw OBU Data')
                        data_txt = st.text('Loading 30-minutes data...')
                        df = data_reader(path)

                        fig, ax = plt.subplots()
                        arr = df.mean(axis=0).iloc[1:].values
                        ax.hist(arr, bins=200)
                        st.pyplot(fig)

                        data_txt.text('Loading 30-minutes data...done!')

                with col1:
                        # select meaninful streets (with variance) and perform feature engineering
                        st.subheader('Select Streets')
                        feat_txt = st.text('Feature Engineering...')
                        df_new, lst_streets, timestamp = feat_engin(mean_value, df)
                        
                        fig, ax = plt.subplots()
                        arr = df_new.mean(axis=0).iloc[:-4].values
                        ax.hist(arr, bins=200)
                        st.pyplot(fig)

                with col2:

                        st.subheader('Add Features')
                        st.text(list(df_new.columns[-4:])[0])
                        st.text(list(df_new.columns[-4:])[1])
                        st.text(list(df_new.columns[-4:])[2])
                        st.text(list(df_new.columns[-4:])[3])
                        

                        feat_txt.text('Feature Engineering...done!')


        st.markdown("""---""")

        with st.beta_container():
                # split and scale the data
                train, val, test, scaler, timestamp_test = data_split_scale(df_new, validation_period, testing_period, n_feat_time, timestamp)

                
                st.subheader('Distributions')
                col4, col5, col6 = st.beta_columns(3)
                
                with col4:               
                        st.subheader('Training set')
                        plot_distribution(train, 'blue')
                                      
                with col5:
                        st.subheader('Validation set')
                        plot_distribution(val, 'orange')

                with col6:
                        st.subheader('Testing set')
                        plot_distribution(test, 'green')

                st.subheader('Splitting')

                col7,  col9 = st.beta_columns((2, 2))
                
                with col7:
                        st.subheader('Pie Chart')
                        plot_pie(train, val, test)
                
                with col9:
                        st.subheader('Line Plot')
                        x = list(range(df_new.shape[0]))
                        line_tr = train[:,:-4].mean(axis=1)
                        line_vl = val[:,:-4].mean(axis=1)
                        line_ts = test[:,:-4].mean(axis=1)
                        labels = ['train','validation','test']

                        fig, ax = plt.subplots()
                        ax.plot(x[:train.shape[0]], line_tr, color='blue')
                        ax.plot(x[train.shape[0]:train.shape[0]+val.shape[0]], line_vl, color='orange')
                        ax.plot(x[train.shape[0]+val.shape[0]:train.shape[0]+val.shape[0]+test.shape[0]], line_ts, color='green')
                        ax.legend(labels)
                        st.pyplot(fig, width =10) 

                st.markdown("""---""")

                # transform the data to tensors
                tensor_train = data_loader(train, inp_sqc, out_sqc, aux_dim, batch_tr)
                tensor_valid = data_loader(val, inp_sqc, out_sqc, aux_dim, batch_vl)
                tensor_test = data_loader(test, inp_sqc, out_sqc, aux_dim, batch_ts)
                logging.info("-- prepare pipeline for tf")


                if st.button('configure the model') or session_state.check1:
                        st.subheader('Configure')

                        hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta = slider_display()
                        
                        save_updates_yaml(hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta)

                        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)
                        
                        session_state.check1 = True

                        if st.button("CONTINUE - TRAIN"):
                                st.subheader('Training')
                                # start training
                                step_epoch = len(train) // batch_tr
                                
                                lstm_ed = training(lstm_ed, nb_epchs, step_epoch, # model, number of epochs, steps per epoch
                                        tensor_train,  tensor_valid, # training and validation tensors
                                        loss_fct, valid_loss_fct, opt, # loss functions and optimizer
                                        patience, delta) # early stopping

                                st.text("Save trained model...")
                                session_state.check1 = False

                                if st.button("RESTART APP TO TEST"):
                                        st.text("Restarting...")

                elif st.button('train with default'):
                        hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta= upload_yaml()
                        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)
                        st.subheader('Training')
                        # start training
                        step_epoch = len(train) // batch_tr
                        
                        lstm_ed = training(lstm_ed, nb_epchs, step_epoch, 
                                                tensor_train,  tensor_valid, # training and validation tensors
                                                loss_fct, valid_loss_fct, opt, patience, delta) # early stopping

                        st.text("Save trained model...")
                        session_state.check1 = False

                        if st.button("RESTART APP TO TEST"):
                                st.text("Restarting...")


                elif st.button('start predictions'):

                        hidd_dim, nb_epchs, rcr, krnl, dr, patience, delta= upload_yaml()
                        lstm_ed = LSTM_ED(total_dim, hidd_dim, rcr, krnl, dr)

                        checkpoint = tf.train.Checkpoint(model = lstm_ed)
                        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

                        st.subheader('Testing')
                        pred, targ, rmse, mae = testing(lstm_ed, tensor_test, aux_dim, scaler, out_sqc, lst_streets, streets, timestamp_test[inp_sqc:]) 

        # logging.info("Finally, I can eat my pizza(s)")

if __name__ == "__main__":
        main()