import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def plot_distribution(df, c):
    fig, ax = plt.subplots()
    arr = df.mean(axis=0)[:-4]
    ax.hist(arr, bins=200,  color=c)
    return st.pyplot(fig)



def plot_pie(train, val, test):
    tot = train.shape[0] +  val.shape[0]+ test.shape[0]
    percentages = [train.shape[0]*100/tot, val.shape[0]*100/tot, test.shape[0]*100/tot]
    labels = ['train','validation','test']           
    fig, ax = plt.subplots()
    ax.pie(percentages,  autopct='%2.1f%%')
    ax.legend(labels)
    return st.pyplot(fig)