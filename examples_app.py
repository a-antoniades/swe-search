import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from moatless.streamlit.tree_vizualization import update_visualization
from moatless.trajectory import Trajectory

# Import the generate_report function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('moatless').setLevel(logging.INFO)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

load_dotenv()

file_path = "moatless/benchmark/swebench_lite_all_evaluations.json"
with open(file_path) as f:
    instances = json.load(f)
    instances_by_id = {instance["instance_id"]: instance for instance in instances}

st.set_page_config(layout="wide", page_title="Moatless Visualizer", initial_sidebar_state="collapsed")


def get_trajectory(trajectory_path: str) -> Trajectory:
    return Trajectory.load(trajectory_path, skip_workspace=True)


def get_example_trajectories():
    examples_dir = "examples"
    trajectories = []
    for root, dirs, files in os.walk(examples_dir):
        if "trajectory.json" in files:
            trajectory_name = os.path.basename(root)
            trajectory_path = os.path.join(root, "trajectory.json")
            trajectories.append((trajectory_name, trajectory_path))
    return trajectories


if __name__ == "__main__":

    if "selected_trajectory_path" not in st.session_state:
        st.session_state.selected_trajectory_path = None

    with st.sidebar:
        st.header("Trajectory Selection")
        example_trajectories = get_example_trajectories()
        selected_trajectory = st.selectbox(
            "Select a trajectory:",
            options=[name for name, _ in example_trajectories],
            format_func=lambda x: x
        )
        
        if st.button("Load Selected Trajectory"):
            selected_path = next(path for name, path in example_trajectories if name == selected_trajectory)
            st.session_state.selected_trajectory_path = selected_path
            st.query_params.update({"trajectory_path": selected_path})
            st.rerun()


    if st.session_state.selected_trajectory_path:
        with st.spinner(f"Loading trajectory: {st.session_state.selected_trajectory_path}"):
            st.session_state.trajectory = get_trajectory(st.session_state.selected_trajectory_path)

        st.header(st.session_state.trajectory.info["instance_id"])
        container = st.container()
        update_visualization(container)
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Select a trajectory:")
            for name, path in example_trajectories:
                if st.button(name, key=f"button_{name}"):
                    st.session_state.selected_trajectory_path = path
                    st.query_params.update({"trajectory_path": path})
                    st.rerun()

    # Sync session state with query params on page load
    if "trajectory_path" in st.query_params and st.session_state.selected_trajectory_path != st.query_params["trajectory_path"]:
        st.session_state.selected_trajectory_path = st.query_params["trajectory_path"]
        st.rerun()
