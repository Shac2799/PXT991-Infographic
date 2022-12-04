import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="animation test", page_icon=None, 
                   layout="wide", initial_sidebar_state="auto", menu_items=None) 

st.title("Animation test!")


df = px.data.gapminder()

fig = px.bar(df, x="continent", y="pop", color="continent",
  animation_frame="year", animation_group="country", range_y=[0,4000000000])
fig.show()

fig.show()


st.pyplot(fig=None, clear_figure=None)
st.set_option('deprecation.showPyplotGlobalUse', False)
