import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from models.simpleRegression import SimpleNeuralNetwork
from d3GraphComponenet import d3_graph

st.set_page_config(page_title="NN Regression", page_icon="📈")

'''# Neural Network Regression Through Gradient Descent Visualization'''
st.divider()

#Prepare state for whenever page first mounts
if 'model_configuration' not in st.session_state:
    st.session_state.model_configuration = {'num_outputs' : 2, 'num_inputs' : 2, 'batch_size': 1, 'layer1_size' : 3, 'layer2_size' : 3, 'activation' : 'Sigmoid', 'is_visible' : False}
    
#Function for updating state and clearing model
def set_model_configuration(num_outputs, num_inputs, batch_size, layer1_size, layer2_size, activation_option, is_visible):
    st.session_state.clear()
    st.session_state.model_configuration = {'num_outputs' : num_outputs, 'num_inputs' : num_inputs, 'batch_size': batch_size, 'layer1_size' : layer1_size, 'layer2_size' : layer2_size, 'activation' : activation_option, 'is_visible' : is_visible}

#Helper function to create model with desired params
def load_model():
    _num_outputs, _num_inputs, _batch_size, _layer1_size, _layer2_size, _activation_option, _is_visible = st.session_state.model_configuration.values()
    return SimpleNeuralNetwork(num_inputs=_num_inputs, hidden_layers=[_layer1_size, _layer2_size], num_outputs= _num_outputs, activation=_activation_option)

#Create some testing data that is formatted to the correct input size
def generate_data():
    inputs = st.session_state.model_configuration['num_inputs']
    outputs = st.session_state.model_configuration['num_outputs']

    sin_x = np.array(np.linspace(start=0, stop=10, num=inputs))
    sin_y = np.array(np.sin((1/5) * 2 * np.pi * sin_x))
    
    ref_sin_x = np.array(np.linspace(start=0, stop=10, num=100))
    ref_sin_y = np.array(np.sin((1/5) * 2 * np.pi * ref_sin_x))
    referenceData = [[x, y] for x, y in zip(ref_sin_x, ref_sin_y)]
    
    return sin_x, sin_y, referenceData
    

#Input form for configuring parameters
with st.form(key='parameters'):
    '''## Configure Training Parameters'''
    num_outputs = st.slider("Number of Outputs from Network", min_value=2, max_value=100, step=1)
    num_inputs = num_outputs
    batch_size = st.slider("Number of Inputs Within Each Batch", min_value=2, max_value=10, step=1)
    st.divider()

    '''## Configure Layers'''
    layer1_size = st.slider("Number of Neurons in First Layer", min_value=1, max_value=10, step=1)
    layer2_size = st.slider("Number of Neurons in Second Layer", min_value=1, max_value=10, step=1)
    activation_option = st.selectbox('Please Select The Activation All Layers Will Inherit', ('Sigmoid', 'Tanh'))
    st.divider()
    
    if st.form_submit_button(label='Generate New Model With Given Parameters'):
        set_model_configuration(num_outputs, num_inputs, batch_size, layer1_size, layer2_size, activation_option, True)

#Track the state of the current model and update data
if 'model' not in st.session_state:
    st.session_state.model = load_model()
test_x, target_y, reference_data = generate_data() 

#Window in which network graph appears and some basic styling
st.divider()
if st.session_state.model_configuration['is_visible']:
    
    with st.container():
        st.header('Model Validation')
        st.divider()
        _batch_size = st.session_state.model_configuration['batch_size']
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            #Implementation of buttons using returned boolean values to avoid on click event handlers from being called at wrong time b/c streamlit
            if st.button('Train 10 epochs'):
                st.session_state.model.train(data=[test_x], target_label=[target_y], batch_size = _batch_size, epochs=10, shuffle= True, learning_rate=0.05)
    
        with col3:
            if st.button('Train 100 epochs'):
                st.session_state.model.train(data=[test_x], target_label=[target_y], batch_size = _batch_size, epochs=100, shuffle= True, learning_rate=0.05)
            
        prediction = st.session_state.model.predict(test_x)
        prediction_data = [[x, y] for x, y in zip(test_x, prediction)]
        d3_graph(data=prediction_data, referenceFunction=reference_data, key=2)
        
        with st.expander("View Predicted Values"):
            data_frame = pd.DataFrame(prediction_data, columns=['x', 'y'])
            st.dataframe(data_frame, use_container_width=True)
        
        st.divider()
    

with st.expander("View Data Set"):
    st.latex(r'''\textrm{Data set contains 1000 linearly spaced points} \\ \textrm{within the closed region}\; \{ sin(x) \; | \; 0 \leq x \leq 10 \}''')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 







    