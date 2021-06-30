import numpy as np
import time
import pandas as pd

from util import logging
import tensorflow as tf
from testing.util import inverse_transform, evaluation_fct

import streamlit as st
import altair as alt
from testing.app_util import update_layer_deck, initial_layer_deck
import pydeck as pdk


def testing(model, tensor_test, aux_dim, scaler, out_sqc, lst, streets, timestamp):

    logging.info('Testing started')
    forecasts = []
    targets = []
    rmse_list = []
    mae_list = []
    mean_rmse = []
    mean_mae = []


    timestamp_txt = st.empty()

    # bar = st.progress(0)
    r = initial_layer_deck()
    map = st.pydeck_chart(r)

    df = pd.DataFrame({'timestamp':[], 'avgPred': [], 'avgTarg': []})
    line = alt.Chart(df, title="LSTM encoder decoder predictions").transform_fold(['avgPred', 'avgTarg']).mark_line().encode(x='timestamp:T',y='value:Q',color='key:N').properties(width=1100, height=500)
    chart = st.altair_chart(line)

    col3, col4 = st.beta_columns(2)

    with col3:
            df_rmse = pd.DataFrame({'timestamp':[],'RMSE': []})
            c = alt.Chart(df_rmse, title="RMSE metric").mark_line().encode(x ='timestamp:T',y='RMSE',  tooltip=['RMSE']).properties(width=500, height=200)
            chart_rmse = st.altair_chart(c, use_container_width=True)

                        
    with col4:
            df_mae = pd.DataFrame({'timestamp':[],'MAE': []})
            c = alt.Chart(df_mae, title="MAE metric").mark_line().encode(x ='timestamp:T', y='MAE',  tooltip=['MAE']).properties(width=500, height=200)
            chart_mae = st.altair_chart(c, use_container_width=True)
            
            

    
    
    for (step, (inp_tot, targ)) in enumerate(tensor_test):

            timestamp_txt.subheader(timestamp.iloc[step])
            
            inp, aux = inp_tot[0], inp_tot[1]
            targ = tf.cast(targ, tf.float32)
            pred = model(inp, aux, training=False)
            
            truth = inverse_transform(targ[0][:,:- aux_dim],  scaler)
            pred = inverse_transform(pred[0][:,:-aux_dim],  scaler)
  
            r = update_layer_deck(lst, streets, pred)
            r.update()

            map.pydeck_chart(r)
                      
            forecasts.append(pred)
            targets.append(truth)
            
            rmse, mae = evaluation_fct(targets, forecasts, out_sqc)

            logging.info(' -- step '+ str(step)+' mae: ' +str(np.mean(mae))+' rmse: '+str(np.mean(rmse)))
            
            rmse_list.append(rmse)
            mae_list.append(mae)

            mean_pred = np.mean(pred[0])
            mean_truth = np.mean(truth[0])
            mean_rmse = np.mean(rmse_list)
            maen_mae = np.mean(mae_list)

            df= pd.DataFrame({'timestamp':[timestamp.iloc[step]], 'avgPred': [mean_pred], 'avgTarg': [mean_truth]})
            chart.add_rows(df)
            
            with col3:
                    df_rmse = pd.DataFrame({'timestamp':[timestamp.iloc[step]],'RMSE': [mean_rmse]})
                    chart_rmse.add_rows(df_rmse)
            with col4:
                    df_mae = pd.DataFrame({'timestamp':[timestamp.iloc[step]], 'MAE': [maen_mae]})
                    chart_mae.add_rows(df_mae)

            

            time.sleep(1)
            # bar.progress(step)

    return forecasts, targets, rmse_list, mae_list