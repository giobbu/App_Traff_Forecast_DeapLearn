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
    target_plot = []
    rmse_list = []
    mae_list = []
    mean_rmse = []
    mean_mae = []


    timestamp_txt = st.empty()
    multi_timestamp_txt = st.empty()

    # bar = st.progress(0)
    r = initial_layer_deck()
    map = st.pydeck_chart(r)


    chart_all = st.empty()
    chart_multi = st.empty()

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

            inp, aux = inp_tot[0], inp_tot[1]
            targ = tf.cast(targ, tf.float32)
            pred = model(inp, aux, training=False)
            
            past = inverse_transform(inp[0][:,:- aux_dim],  scaler)
            truth = inverse_transform(targ[0][:,:- aux_dim],  scaler)
            pred = inverse_transform(pred[0][:,:-aux_dim],  scaler)
  
            r = update_layer_deck(lst, streets, pred)
            r.update()

            map.pydeck_chart(r)
                      
            forecasts.append(pred)
            targets.append(truth)

            if step == 0:
                target_plot.append(past)
            else:
                target_plot.append(past[-1])
            
            rmse, mae = evaluation_fct(targets, forecasts, out_sqc)

            logging.info(' -- step '+ str(step)+' mae: ' +str(np.mean(mae))+' rmse: '+str(np.mean(rmse)))
            
            rmse_list.append(rmse)
            mae_list.append(mae)

            mean_pred_multi = np.sum(pred, axis=1)
            mean_truth_multi = np.sum(truth, axis=1)

            all_truth = np.sum(np.vstack(target_plot), axis=1)

            mean_rmse = np.mean(rmse_list)
            maen_mae = np.mean(mae_list)

            df_all = pd.DataFrame({'timestamp':timestamp.iloc[:step+12], 'Targ': all_truth})
            line_all = alt.Chart(df_all).transform_fold(['Targ']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N').properties(width=800, height=500)

            df_multi_targ = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Targ_': mean_truth_multi})
            line_multi_targ = alt.Chart(df_multi_targ).transform_fold(['Targ_']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N').properties(width=800, height=500)
            
            df_multi_pred = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Pred': mean_pred_multi})
            line_multi_pred = alt.Chart(df_multi_pred).transform_fold(['Pred']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N').properties(width=800, height=500)

            chart_all.altair_chart(line_all + line_multi_pred + line_multi_targ)
        

            df_multi = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Pred': mean_pred_multi, 'Targ': mean_truth_multi})
            line_multi = alt.Chart(df_multi, title = 'Prediction from ' +str(timestamp.iloc[step])+' to '+str(timestamp.iloc[step+11])).transform_fold(['Pred', 'Targ']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N').properties(width=800, height=500)
            chart_multi.altair_chart(line_multi)
            
            with col3:
                    df_rmse = pd.DataFrame({'timestamp':[timestamp.iloc[step]],'RMSE': [mean_rmse]})
                    chart_rmse.add_rows(df_rmse)

            with col4:
                    df_mae = pd.DataFrame({'timestamp':[timestamp.iloc[step]], 'MAE': [maen_mae]})
                    chart_mae.add_rows(df_mae)

            

            

        #     time.sleep(1)
            # bar.progress(step)

    return forecasts, targets, rmse_list, mae_list