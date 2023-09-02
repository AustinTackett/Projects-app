import streamlit.components.v1 as components
from typing import List
from typing import Tuple
import os

#Tell streamlit to look at port 3001 to find react component in pipeline
_my_component_func = components.declare_component(
    "d3_graph",
    url="http://localhost:3001/")

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend/build")
component = components.declare_component("d3_graph", path=build_dir)

def d3_graph(data: List[Tuple[float, float]], referenceFunction: List[Tuple[float, float]], key=None):
    component_value = _my_component_func(data=data, referenceFunction=referenceFunction, key=key, default=0)
    return component_value

    