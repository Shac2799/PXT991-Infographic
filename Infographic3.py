from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

st.title("Infographic Showing Effect of Mass on Gravitational Strength")

Body_size = st.slider("Diameter of Celestial Body [km]", 1, 10000, 5000)



