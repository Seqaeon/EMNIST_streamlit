import os
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit_analytics2
from PIL import Image

import data_prep as data


(MN_TRAIN, MN_TRAIN_Z), (MN_TEST, MN_TEST_Z) = data.choose_data(dataset=chosen_set)

st.write(MN_TRAIN_Z[1])
st.write("Sucessful!!")
