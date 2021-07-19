from altair.vegalite.v4.schema.channels import Opacity, StrokeDash
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
            chart_errorrmse_multi = st.empty()

            df_rmse = pd.DataFrame({'timestamp':[],'RMSE': []})
            c = alt.Chart(df_rmse).mark_line().encode(x ='timestamp:T',y='RMSE',  tooltip=['RMSE']).properties(width=500, height=200)
            chart_rmse = st.altair_chart(c, use_container_width=True)

                        
    with col4:
            chart_errormae_multi = st.empty()

            df_mae = pd.DataFrame({'timestamp':[],'MAE': []})
            c = alt.Chart(df_mae).mark_line().encode(x ='timestamp:T', y='MAE',  tooltip=['MAE']).properties(width=500, height=200)
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

            mean_rmse_multi = np.mean(rmse, axis=1)
            mean_stdrmse_multi = np.std(rmse, axis=1)

            mean_mae_multi = np.mean(mae, axis=1)
            mean_stdmae_multi = np.std(mae, axis=1)
            
            with col3:
                df_multi_rmse = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'RMSE': mean_rmse_multi})
                df_multi_rmse_std_pos = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'RMSE': mean_rmse_multi + mean_stdrmse_multi})
                df_multi_rmse_std_neg = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'RMSE': mean_rmse_multi - mean_stdrmse_multi})

                multi_rmse_ci_pos = alt.Chart(df_multi_rmse_std_pos).transform_fold(['RMSE']).mark_line(opacity = 0.3,  color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                multi_rmse_ci_neg = alt.Chart(df_multi_rmse_std_neg).transform_fold(['RMSE']).mark_line(opacity = 0.3, color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)

                dot_multi_rmse = alt.Chart(df_multi_rmse).transform_fold(['RMSE']).mark_point(filled=True, color='black').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                line_multi_rmse = alt.Chart(df_multi_rmse).transform_fold(['RMSE']).mark_line(opacity = 0.3, color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                
                chart_errorrmse_multi.altair_chart(multi_rmse_ci_pos +line_multi_rmse + dot_multi_rmse + multi_rmse_ci_neg, use_container_width=True)

            with col4:
                df_multi_mae = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'MAE': mean_mae_multi})
                df_multi_mae_std_pos = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'MAE': mean_mae_multi + mean_stdmae_multi})
                df_multi_mae_std_neg = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'MAE': mean_mae_multi - mean_stdmae_multi})

                multi_mae_ci_pos = alt.Chart(df_multi_mae_std_pos).transform_fold(['MAE']).mark_line(opacity = 0.3,  color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                multi_mae_ci_neg = alt.Chart(df_multi_mae_std_neg).transform_fold(['MAE']).mark_line(opacity = 0.3, color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)

                dot_multi_mae = alt.Chart(df_multi_mae).transform_fold(['MAE']).mark_point(filled=True, color='black').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                line_multi_mae = alt.Chart(df_multi_mae).transform_fold(['MAE']).mark_line(opacity = 0.3, color='red').encode(x='timestamp:T', y='value:Q').properties(width=300, height=200)
                
                chart_errormae_multi.altair_chart(multi_mae_ci_pos +line_multi_mae + dot_multi_mae + multi_mae_ci_neg, use_container_width=True)


            df_all = pd.DataFrame({'timestamp':timestamp.iloc[:step+12], 'Targ': all_truth})
            line_all = alt.Chart(df_all, title = 'Prediction from ' +str(timestamp.iloc[step])+' to '+str(timestamp.iloc[step+11])).transform_fold(['Targ']).mark_line(color='red').encode(x='timestamp:T', y='value:Q').properties(width=800, height=500)

            df_multi_targ = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Targ_': mean_truth_multi})
            line_multi_targ = alt.Chart(df_multi_targ).transform_fold(['Targ_']).mark_line().encode(x='timestamp:T', y='value:Q', color= 'key:N').properties(width=800, height=500)
            
            df_multi_pred = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Pred': mean_pred_multi})
            line_multi_pred = alt.Chart(df_multi_pred).transform_fold(['Pred']).mark_line().encode(x='timestamp:T', y='value:Q',  color='key:N').properties(width=800, height=500)

            chart_all.altair_chart(line_all + line_multi_pred + line_multi_targ)
        
            df_multi = pd.DataFrame({'timestamp':timestamp.iloc[step+12:step+24], 'Pred': mean_pred_multi, 'Targ': mean_truth_multi})
            line_multi = alt.Chart(df_multi, title = 'Zoom on Prediction').transform_fold(['Pred', 'Targ']).mark_line().encode(x='timestamp:T', y='value:Q',color='key:N').properties(width=800, height=500)
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