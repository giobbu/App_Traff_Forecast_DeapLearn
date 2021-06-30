
import streamlit as st
import yaml

with open('config.yaml') as file:
        config = yaml.safe_load(file)

path = config['script_path']

HIDD_DIM = config['model']['hidd_dim']
NB_EPOCHS = config['model']['nb_epochs']
RCR = config['model']['rcr_init']
KRNL = config['model']['krnl_reg']
DR = config['model']['dropout']
PATIENCE = config['tr_phs']['patience']
DELTA = config['tr_phs']['min_delta']

def slider_display():
    hidd_dim  = st.sidebar.selectbox('Hidden Dimension',(HIDD_DIM, 50, 75))
    nb_epochs  = st.sidebar.selectbox('Number of Epochs',(1, 5, NB_EPOCHS, 50, 75))
    rcr  = st.sidebar.selectbox('Recurrent Initializer',(RCR, 'HeNormal', 'GlorotNormal'))
    krnl  = st.sidebar.selectbox('Kernel Regulizer',(KRNL, 0.002, 0.003))
    dr  = st.sidebar.selectbox('Dropout',(DR, 0.2, 0.3))
    patience  = st.sidebar.selectbox('EarlyStop - Patience',(10, PATIENCE, 30))
    delta  = st.sidebar.selectbox('EarlyStop - Delta',(0.00005, DELTA, 0.0002))


    return hidd_dim, nb_epochs, rcr, krnl, dr, patience, delta
                        

def save_updates_yaml(hidd_dim, nb_epochs, rcr, krnl, dr, patience, delta):
    
    config['model']['hidd_dim'] = hidd_dim
    config['model']['nb_epochs'] = nb_epochs
    config['model']['rcr_init'] =  rcr
    config['model']['krnl_reg'] = krnl
    config['model']['dropout'] = dr
    config['tr_phs']['patience'] = patience
    config['tr_phs']['min_delta'] = delta

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)


def upload_yaml():
        HIDD_DIM = config['model']['hidd_dim']
        NB_EPOCHS = config['model']['nb_epochs']
        RCR = config['model']['rcr_init']
        KRNL = config['model']['krnl_reg']
        DR = config['model']['dropout']
        PATIENCE = config['tr_phs']['patience']
        DELTA = config['tr_phs']['min_delta']
        return HIDD_DIM, NB_EPOCHS, RCR, KRNL, DR, PATIENCE, DELTA
