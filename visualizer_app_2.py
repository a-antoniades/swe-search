import argparse
import json
import logging
import os
import sys
from typing import Tuple

import streamlit as st
from dotenv import load_dotenv

from moatless.benchmark.report_v2 import to_result
from moatless.benchmark.utils import get_moatless_instance
from moatless.edit import EditCode, PlanToCode
from moatless.edit.plan import RequestCodeChange
from moatless.loop import AgenticLoop
from moatless.schema import TestStatus
from moatless.state import AgenticState, Pending, Rejected, Finished
from moatless.trajectory import Trajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('moatless').setLevel(logging.INFO)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

load_dotenv()

file_path = "moatless/benchmark/swebench_lite_all_evaluations.json"
with open(file_path) as f:
    instances = json.load(f)
    instances_by_id = {instance["instance_id"]: instance for instance in instances}

st.set_page_config(layout="wide", page_title="Moatless Visualizer")

print(f"session_state: {st.session_state}")

if "selected_instance_id" not in st.session_state:
    st.session_state.selected_instance_id = None

if "loop" in st.session_state and st.session_state.loop is not None:
    print(f"trajectory: {st.session_state.loop.trajectory}")

def update_visualization():
    
    with st.container():
        if "instance_id" in st.session_state.loop.trajectory.info:
            instance_id = st.session_state.loop.trajectory.info["instance_id"]
            col1, col2 = st.columns([3, 1])
            with col1:
                st.header(f"Instance: {instance_id}")
            with col2:
                if st.session_state.loop.trajectory.info.get("resolved", False):
                    st.success("✅ Resolved")
                else:
                    st.error("❌ Not Resolved")

            tabs = st.tabs(["Current", "Submission"])

            with tabs[0]:
                col1, col2, col3 = st.columns([1, 3, 2])

                with col1:
                    st.subheader("Metrics")
                    metrics = {
                        "Duration": f"{st.session_state.loop.trajectory.info.get('duration', 'N/A'):.2f} s" if isinstance(st.session_state.loop.trajectory.info.get('duration'), (int, float)) else "N/A",
                        "Total Cost": f"${st.session_state.loop.trajectory.info.get('total_cost', 'N/A'):.6f}" if isinstance(st.session_state.loop.trajectory.info.get('total_cost'), (int, float)) else "N/A",
                        "Prompt Tokens": st.session_state.loop.trajectory.info.get('prompt_tokens', 'N/A'),
                        "Completion Tokens": st.session_state.loop.trajectory.info.get('completion_tokens', 'N/A')
                    }
                    for key, value in metrics.items():
                        st.metric(key, value)

                with col2:
                    if st.session_state.loop.trajectory.info.get("error"):
                        st.subheader("Error")
                        st.code(st.session_state.loop.trajectory.info["error"])
                    else:
                        st.subheader("Submission")
                        if "submission" in st.session_state.loop.trajectory.info:
                            st.code(st.session_state.loop.trajectory.info["submission"], language="python")
                        else:
                            st.write("No submission available")

                with col3:
                    st.subheader("Evaluation Result")
                    if "evaluation_result" in st.session_state.loop.trajectory.info:
                        eval_result = st.session_state.loop.trajectory.info["evaluation_result"]
                        tests_status = eval_result.get("tests_status", {})

                        resolution_status = tests_status.get("status", "Unknown")
                        st.info(f"Resolution Status: {resolution_status}")

                        fail_to_pass = tests_status.get("fail_to_pass", {}).get("failure", [])
                        if fail_to_pass:
                            st.subheader("Failed Tests (Fail to Pass)")
                            error_message = "\n".join(f"• {test}" for test in fail_to_pass)
                            st.error(error_message)
                        
                        pass_to_pass = tests_status.get("pass_to_pass", {}).get("failure", [])
                        if pass_to_pass:
                            st.subheader("Failed Tests (Pass to Pass)")
                            error_message = "\n".join(f"• {test}" for test in pass_to_pass)
                            st.error(error_message)
                        
                        if not fail_to_pass and not pass_to_pass:
                            st.success("No tests failed")

                        st.subheader("Full Evaluation Result")
                        st.json(eval_result, expanded=False)

                        with st.expander("Output"):
                            st.code(eval_result.get("output", "No output available"))
                    else:
                        st.write("No evaluation result available")

            with tabs[1]:
                col1, col2, col3 = st.columns([2, 2, 2])
                instance = get_moatless_instance(instance_id)

                with col1:
                    st.subheader("Submission")
                    if "submission" in st.session_state.loop.trajectory.info:
                        st.code(st.session_state.loop.trajectory.info["submission"], language="python")
                    else:
                        st.write("No submission available")

                with col2:
                    st.subheader("Golden patch")
                    st.code(instance["golden_patch"])

                with col3:
                    st.subheader("Test patch")
                    st.code(instance["test_patch"])

        for transition in st.session_state.loop.trajectory.transitions:
            state = transition.state
            if isinstance(state, Pending):
                continue

            # st.session_state.loop.trajectory.restore_from_snapshot(transition)
        
            with st.container():
                st.subheader(f"{state.id}: {state.name}")
                col1, col2 = st.columns([1, 1])
                with col1:
                    tabs = st.tabs(["Context", "File Context", "Prompt"])
                    
                    with tabs[0]:

                        state_info = state.model_dump(exclude_none=True, exclude={"previous_state", "next_states", "value_function_result"})
                        state_info = {"name": state.name, **state_info}
                        if state.previous_state:
                            state_info["previous_state_id"] = state.previous_state.id

                        st.json(state_info, expanded=False)

                        if isinstance(state, Rejected) or isinstance(state, Finished):
                            st.write(f"State: {state.message}")
                        if isinstance(state, AgenticState):
                            st.write(f"Model: {state.model} (temperature: {state.temperature}")

                            if state.completion and state.completion.usage:
                                st.write(f"{state.completion.usage.prompt_tokens} prompt tokens")

                            if isinstance(state, PlanToCode):
                                if state.test_results:
                                    st.subheader("Test results")
                                    failed_tests = [issue for issue in state.test_results if issue.status in [TestStatus.FAILED, TestStatus.ERROR]]
                                    st.write(f"Ran {len(state.test_results)}, {len(failed_tests)} failed")

                                    for issue in failed_tests:
                                        with st.expander(f"{issue.file_path} {issue.span_id}"):
                                            st.write(f"File: {issue.file_path}")
                                            st.write(f"Span ID: {issue.span_id}")
                                            st.code(issue.message, language="bash")

                            if isinstance(state, EditCode) and "<search>" in state.completion.input[-1]["content"]:
                                st.subheader("Search")
                                message = state.completion.input[-1]
                                search_block = message["content"].split("<search>")[-1].split("</search>")[0]
                                st.code(search_block, language="python")


                    with tabs[1]:
                        if isinstance(state, AgenticState):
                            file_context = st.session_state.loop.workspace.file_context
                            st.markdown("""
                                <style>
                                    .file-context-code {
                                        max-height: 300px;
                                        overflow-y: auto;
                                    }
                                </style>
                            """, unsafe_allow_html=True)

                            st.json(file_context.model_dump())

                            st.code(file_context.create_prompt(), language="python")
                            st.markdown('<style>div.stCode > div {max-height: 300px;}</style>', unsafe_allow_html=True)

                    with tabs[2]:
                        if isinstance(state, AgenticState) and state.completion:
                            st.json({
                                "model": state.completion.model,
                                "usage": state.completion.usage.model_dump() if state.completion.usage else None
                            })
                            
                            if state.completion.input:
                                st.subheader("Inputs")
                                for idx, input_msg in enumerate(state.completion.input):
                                    with st.expander(f"Message {idx + 1} by {input_msg['role']}", expanded=(idx == len(state.completion.input) - 1)):
                                        st.json(input_msg)
                            
                        else:
                            st.info("No completion information available for this state.")

                with col2:
                    tabs = st.tabs(["Action", "Completion"])

                    with tabs[0]:
                        if isinstance(state, AgenticState):

                            if isinstance(state, EditCode):
                                if state.action_request and "<replace>" in state.action_request.content:
                                    replace_block = state.action_request.content.split("<replace>")[-1].split("</replace>")[0]
                                    if replace_block:
                                        st.subheader("Replace")
                                        st.code(replace_block, language="python")

                                if state.outcome.get("diff"):
                                    st.subheader("Diff")
                                    st.code(state.outcome["diff"], language="diff")

                                #if state.outcome.trigger == "reject":
                                #    st.warning("Rejected")

                                if state.outcome.get("message"):
                                    st.write(state.outcome["message"])

                            expand_action = isinstance(state, PlanToCode)

                            if state.action_request:
                                st.subheader("Action")
                                action_info = state.action_request.model_dump(exclude_none=True, exclude={"thoughts", "scratch_pad"})
                                st.json(action_info, expanded=expand_action)

                                thoughts = state.action_request.thoughts if hasattr(state.action_request, 'thoughts') else None
                                thoughts = thoughts or (state.action_request.scratch_pad if hasattr(state.action_request, 'scratch_pad') else None)
                                thoughts = thoughts or state.action_request.action.scratch_pad if hasattr(state.action_request, 'action') else None
                            
                                if thoughts:
                                    st.subheader("Thoughts")
                                    st.write(thoughts)

                                if state.response:
                                    st.subheader("Outcome")
                                    outcome_info = state.response.model_dump(exclude_none=True)
                                    st.json(outcome_info, expanded=False)


                    with tabs[1]:
                        if isinstance(state, AgenticState) and state.completion:
                            if state.completion.response:
                                st.subheader("Response")
                                st.json(state.completion.response)

def reset_cache():
    st.cache_data.clear()
    st.session_state.loop = None
    st.session_state.running = False
    st.session_state.selected_node_id = None
    st.info("Cache cleared and session state reset")

def reset_trajectory_cache():
    st.cache_data.clear()
    st.session_state.pop('loop', None)
    st.session_state.pop('running', None)
    st.session_state.pop('selected_trajectory', None)
    st.rerun()

def parse_args():
    parser = argparse.ArgumentParser(description="Moatless Visualizer")
    parser.add_argument("--moatless_dir", default=os.getenv("MOATLESS_DIR", "/tmp/moatless"),
                        help="Directory for Moatless data (default: $MOATLESS_DIR or /tmp/moatless)")
    
    if '--' in sys.argv:
        # Remove streamlit-specific arguments
        streamlit_args = sys.argv[sys.argv.index('--') + 1:]
        args = parser.parse_args(sys.argv[1:sys.argv.index('--')])
    else:
        args = parser.parse_args()
    
    return args

def get_loop(name: str, moatless_dir: str = os.getenv("MOATLESS_DIR", "/tmp/moatless")) -> AgenticLoop:
    repo_from_dir_name = name.split("/")[0]
    if repo_from_dir_name:
        repo_path = f"/tmp/repos/{repo_from_dir_name}"
    else:
        repo_path = None

    trajectory_path = os.path.join(moatless_dir, name, "trajectory.json")
    return AgenticLoop.from_trajectory_file(trajectory_path, base_repo_dir="/tmp/repos", repo_path=repo_path)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_trajectories(moatless_dir: str = os.getenv("MOATLESS_DIR", "/tmp/moatless")) -> list[Tuple[Trajectory, str]]:
    trajectories = []
    for root, _, files in os.walk(moatless_dir):
        if "trajectory.json" in files:
            trajectory_path = os.path.join(root, "trajectory.json")
            try:
                rel_path = os.path.relpath(root, moatless_dir)
                trajectory = Trajectory.load(trajectory_path, skip_workspace=True)
                trajectories.append((trajectory, rel_path))
            except Exception as e:
                logger.error(f"Failed to load trajectory from {trajectory_path}: {e}")

    trajectories.sort(key=lambda x: x[0].info.get("instance_id", "Unknown"))
    return trajectories

if __name__ == "__main__":
    args = parse_args()
    moatless_dir = args.moatless_dir

    st.sidebar.text(f"Using Moatless directory: {moatless_dir}")

    st.sidebar.header("Trajectories")
    trajectories = get_trajectories(moatless_dir)

    for trajectory, trajectory_path in trajectories:
        instance_id = trajectory.info.get("instance_id", "Unknown")
        resolved = "✅" if trajectory.info.get("resolved", False) else "❌"
        if st.sidebar.button(f"{instance_id} - {resolved}", key=f"trajectory_{instance_id}"):
            st.session_state.clear()
            st.query_params["trajectory_id"] = trajectory_path
            st.rerun()

    st.sidebar.header("Cache Management")
    if st.sidebar.button("Reset Trajectory Cache"):
        reset_trajectory_cache()

    if "trajectory_id" in st.query_params:
        selected_trajectory = st.query_params["trajectory_id"]
        if "loop" not in st.session_state:
            with st.spinner(f"Loading trajectory: {selected_trajectory}"):
                st.session_state.loop = get_loop(selected_trajectory, moatless_dir)
                st.query_params["trajectory_id"] = selected_trajectory
                st.session_state.running = False
                st.session_state.selected_trajectory = selected_trajectory

    if "loop" in st.session_state:
        update_visualization()
    else:
        st.header("Trajectory List")
        
        # Create a list to store the data for the dataframe
        trajectory_data = []
        
        for trajectory, trajectory_path in trajectories:
            info = trajectory.info
            instance_id = info.get("instance_id", "Unknown")
            resolved = info.get("resolved", False)
            instance = get_moatless_instance(instance_id)
            result = to_result(instance, trajectory)

            if resolved:
                status = "resolved"
                icon = "✅"
            elif isinstance(trajectory.get_current_state(), Rejected):
                status = "rejected"
                icon = "🚫"
            elif info.get("error"):
                status = "error"
                icon = "❗"
            elif info.get("status") == "finished":
                status = "failed"
                icon = "❌"
            else:
                status = "running"
                icon = "🔄"
            
            duration = f"{info.get('duration', 'N/A'):.2f} s" if isinstance(info.get('duration'), (int, float)) else "N/A"
            total_cost = f"${info.get('total_cost', 'N/A'):.6f}" if isinstance(info.get('total_cost'), (int, float)) else "N/A"

            trajectory_data.append({
                "Instance ID": f"{icon} {instance_id}",
                "Status": status,
                "Duration": duration,
                "Total Cost": total_cost,
                "Transitions": len(trajectory.transitions),
                "Coding status": result.coding.status,
                "Retried plans": result.coding.plan_retries,
                "Retried edits": result.coding.edit_retries,
                "Rejected edits": result.coding.rejected,
                "Path": trajectory_path
            })
        
        # Create a dataframe and display it
        import pandas as pd
        df = pd.DataFrame(trajectory_data)
        
        # Add filtering options
        st.subheader("Filter Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect("Status", df["Status"].unique())
            has_retries_filter = st.multiselect("Has Retries", ["Yes", "No"])
        with col2:
            coding_status_filter = st.multiselect("Coding Status", df["Coding status"].unique())
            has_rejections_filter = st.multiselect("Has Rejections", ["Yes", "No"])
        with col3:
            instance_filter = st.text_input("Instance ID (contains)")

        # Apply filters
        mask = pd.Series(True, index=df.index)
        if status_filter:
            mask &= df["Status"].isin(status_filter)
        if coding_status_filter:
            mask &= df["Coding status"].isin(coding_status_filter)
        if instance_filter:
            mask &= df["Instance ID"].apply(lambda x: instance_filter.lower() in x.lower())
        if has_retries_filter:
            has_retries = df["Retried plans"] > 0 | df["Retried edits"] > 0
            if "Yes" in has_retries_filter:
                mask &= has_retries
            if "No" in has_retries_filter:
                mask &= ~has_retries
        if has_rejections_filter:
            has_rejections = df["Rejected edits"] > 0
            if "Yes" in has_rejections_filter:
                mask &= has_rejections
            if "No" in has_rejections_filter:
                mask &= ~has_rejections

        filtered_df = df[mask]

        # Create a column with clickable links
        filtered_df['Select'] = filtered_df.apply(lambda row: f'<a href="?trajectory_id={row["Path"]}">View</a>', axis=1)
        
        # Display the filtered dataframe with the clickable column
        st.write(filtered_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Add CSS to style the table
        st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {background-color: #f5f5f5;}
        </style>
        """, unsafe_allow_html=True)
