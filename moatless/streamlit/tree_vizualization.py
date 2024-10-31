import logging
import os
from io import BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

from moatless.benchmark.mcts_vizualization_model import build_graph
from moatless.state import AgenticState


def decide_badge(node_info):
    """
    Decide which badge to show for a node based on its properties and the instance data.
    Returns a tuple of (symbol, color) or None if no badge should be shown.
    """
    if node_info.get("resolved") is not None:
        if node_info.get("resolved", False):
            return ('star', 'gold')
        else:
            return ('x', 'red')

    if node_info.get("rejected"):
        return ('x', 'red')

    if node_info.get("no_diff"):
        return ('square', 'yellow')

    if node_info.get("failed_tests") or node_info.get("error_tests"):
        return ('square', 'red')

    if node_info.get("verification_errors"):
        return ('diamond', 'yellow')

    if node_info.get("expected_span_identified"):
        return ('star', 'gold')

    if node_info.get("resolved") is not None:
        if node_info.get("resolved", False):
            return ('star', 'gold')
        else:
            return ('x', 'red')

    if node_info.get("file_identified"):
        return ('circle', 'green')

    if node_info.get("alternative_span_identified"):
        return ('triangle-up', 'blue')

    return None


def update_visualization(container):
    container.empty()
    with container:
        graph_col, info_col = st.columns([6, 3])
        with graph_col:
            G = build_graph(st.session_state.trajectory)
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

            selected_transition_ids = []
            current_state = st.session_state.trajectory.get_current_state()
            while current_state:
                selected_transition_ids.append(current_state.id)
                current_state = current_state.previous_state

            # Create a mapping of point indices to node IDs
            point_to_node_id = {i: node for i, node in enumerate(G.nodes())}

            # Normalize positions to fit in [0, 1] range
            x_values, y_values = zip(*pos.values())
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            # Calculate the scaling factor based on the number of nodes
            num_nodes = len(G.nodes())
            height_scale = max(1, num_nodes / 20)  # Increase height as nodes increase

            for node in pos:
                x, y = pos[node]
                normalized_x = 0.5 if x_max == x_min else (x - x_min) / (x_max - x_min)
                normalized_y = 0.5 if y_max == y_min else (y - y_min) / (y_max - y_min)
                pos[node] = (normalized_x, normalized_y * height_scale)

            edge_x, edge_y = [], []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x, node_y = [], []
            node_colors, node_sizes, node_text, node_labels = [], [], [], []
            badge_x, badge_y, badge_symbols, badge_colors = [], [], [], []
            node_line_widths = []
            node_line_colors = []

            for node in G.nodes():
                node_info = G.nodes[node]
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                reward = None

                if node_info.get('name') == "Rejected":
                    node_colors.append('red')
                elif node_info.get('type') == "action":
                    node_colors.append('#6c7aaa')  # Grayish blue
                elif node_info.get('visits', 0) == 0:
                    node_colors.append('gray')
                else:
                    reward = node_info['value']
                    if reward is None:
                        reward = node_info.get('avg_reward', 0)
                    node_colors.append(reward)

                node_sizes.append(60)

                # Set border width and color based on whether the node is in selected_transition_ids
                if node in selected_transition_ids:
                    node_line_widths.append(4)
                    node_line_colors.append('purple')
                else:
                    node_line_widths.append(2)
                    node_line_colors.append('rgba(0,0,0,0.5)')

                if node_info.get("type") == "node":
                    badge = decide_badge(node_info)
                    if badge:
                        badge_x.append(x + 0.02)  # Offset slightly to the right
                        badge_y.append(y + 0.03)  # Offset slightly up
                        badge_symbols.append(badge[0])
                        badge_colors.append(badge[1])

                    extra = ""
                    if node_info.get("expected_span_identified"):
                        extra = "Expected span was identified"
                    elif node_info.get("file_identified"):
                        extra = "Expected file identified"
                    elif node_info.get("alternative_span_identified"):
                        extra = "Alternative span identified"

                    if node_info.get("no_diff"):
                        extra += "<br>No diff found"

                    if node_info.get("verification_errors"):
                        extra += "<br>Verification errors found"

                    if node_info.get("state_params"):
                        extra += "<br>".join(node_info["state_params"])

                    # Update hover text to include badge information
                    node_text.append(f"ID: {node}<br>"
                                     f"State: {node_info['name']}<br>"
                                     f"Visits: {node_info.get('visits', 0)}<br>"
                                     f"Sum value: {node_info.get('sum_value', 0):.4f}<br>"
                                     f"Value: {node_info.get('value', 0)}<br>"
                                     f"Avg Reward: {node_info.get('avg_reward', 0)}<br>{extra}")

                    if node_info['name'] == "Pending":
                        node_labels.append(f"Start")
                    elif node_info['name'] == "Finished":
                        node_labels.append(f"{node_info['name']}{node_info['id']}<br>{node_info.get('value', '0')}")
                    else:
                        node_labels.append(f"{node}<br>{node_info.get('value', '0')}")

                else:
                    extra = ""
                    if node_info.get("info_params"):
                        extra = "<br>".join(node_info["info_params"])

                    if node_info.get("warnings"):
                        extra += "<br>Warnings:<br>"
                        extra += "<br>".join(node_info["warnings"])

                        badge = ('diamond', 'red')
                        badge_x.append(x + 0.02)  # Offset slightly to the right
                        badge_y.append(y + 0.03)  # Offset slightly up
                        badge_symbols.append(badge[0])
                        badge_colors.append(badge[1])
                    elif node_info.get("expected_span_identified") or node_info.get("alternative_span_identified"):
                        badge = ('star', 'gold')
                        badge_x.append(x + 0.02)  # Offset slightly to the right
                        badge_y.append(y + 0.03)  # Offset slightly up
                        badge_symbols.append(badge[0])

                    node_text.append(extra)

                    node_labels.append(f"{node}<br>{node_info.get('name', 'unknown')}")

            # Create the figure without FigureWidget
            fig = go.Figure(make_subplots())

            # Add edge trace
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))

            # Add node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale=[
                        [0, 'red'],  # -100 to 0
                        [0.25, 'red'],  # -100 to 0
                        [0.5, 'yellow'],  # 0 to 75
                        [1, 'green']  # 75 to 100
                    ],
                    color=node_colors,
                    size=node_sizes,
                    colorbar=dict(
                        thickness=15,
                        title='Avg Reward',
                        xanchor='left',
                        titleside='right',
                        tickmode='array',
                        tickvals=[-100, 0, 50, 75, 100],
                        ticktext=['-100', '0', '50', '75', '100']
                    ),
                    cmin=-100,
                    cmax=100,
                    line=dict(width=node_line_widths, color=node_line_colors)
                ),
                text=node_labels,
                hovertext=node_text,
                textposition='middle center',
            )
            fig.add_trace(node_trace)

            # Add badge trace only if there are badges to show
            if badge_x:
                badge_trace = go.Scatter(
                    x=badge_x, y=badge_y,
                    mode='markers',
                    hoverinfo='none',
                    marker=dict(
                        symbol=badge_symbols,
                        size=10,
                        color=badge_colors,
                        line=dict(width=1, color='rgba(0, 0, 0, 0.5)')
                    ),
                    showlegend=False
                )
                fig.add_trace(badge_trace)

            fig.update_layout(
                title='Trajectory Tree',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600 * height_scale,  # Adjust the height based on the number of nodes
            )

            chart_placeholder = st.empty()
            event = chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False},
                                                   on_select="rerun")

            if event and 'selection' in event and 'points' in event['selection']:
                selected_points = event['selection']['points']
                if selected_points:
                    point_index = selected_points[0]['point_index']
                    if point_index in point_to_node_id:
                        st.session_state.selected_node_id = point_to_node_id[point_index]

            focus_node = st.text_input("Focus on node (optional):", "")
            max_depth = st.number_input("Max depth from focus node:", min_value=1, value=5)
            orientation = st.radio("PDF Orientation", ["landscape", "portrait"])

            focus_node = focus_node if focus_node else None
            generate_pdf = st.button("Generate PDF")
    
        with info_col:
            node_options = [node for node in G.nodes()]

            # Find the index of the current selected node in the options list
            if "selected_node_id" in st.session_state and st.session_state.selected_node_id:
                selected_node_id = st.session_state.selected_node_id
            else:
                selected_node_id = "Node0"

            current_index = next((i for i, option in enumerate(node_options) if option == selected_node_id), 0)

            selected_node_option = st.selectbox("Select Node", node_options, index=current_index,
                                                key="selected_node_option")
            if selected_node_option:
                if selected_node_option.startswith("Node"):
                    st.session_state.selected_node_id = int(selected_node_option.split("Node")[1])
                    st.session_state.selected_type = "node"
                elif selected_node_option.startswith("Action"):
                    st.session_state.selected_node_id = int(selected_node_option.split("Action")[1])
                    st.session_state.selected_type = "action"

            if st.session_state.selected_node_id is not None:
                node_id = st.session_state.selected_node_id

                state = st.session_state.trajectory.get_state(node_id)

                tabs = ["State"]
                snapshot = st.session_state.trajectory.get_snapshot(node_id)
                if snapshot and "file_context" in snapshot and "files" in snapshot["file_context"]:
                    tabs.append("File Context")

                if st.session_state.selected_type == "node" and state.value_function_result:
                    tabs.append("Value Function")

                thoughts = None
                if st.session_state.selected_type == "action" and isinstance(state, AgenticState):
                    if state.action_request:
                        thoughts = state.action_request.thoughts if hasattr(state.action_request, 'thoughts') else None
                        thoughts = thoughts or (
                            state.action_request.scratch_pad if hasattr(state.action_request, 'scratch_pad') else None)
                        if thoughts:
                            tabs.append("Thoughts")
                        tabs.append("Action")
                    if state.response:
                        tabs.append("Outcome")

                    if state.completion:
                        tabs.extend(["Completion"])

                tab_contents = st.tabs(tabs)

                with tab_contents[tabs.index("State")]:
                    state_info = state.model_dump(exclude_none=True,
                                                  exclude={"previous_state", "next_states", "origin_state", "clones",
                                                           "value_function_result"})
                    state_info = {"name": state.name, **state_info}
                    st.json(state_info)

                if "File Context" in tabs:
                    with tab_contents[tabs.index("File Context")]:
                        snapshot = st.session_state.trajectory.get_snapshot(node_id)
                        logging.info(snapshot)

                        if "file_context" in snapshot and "files" in snapshot["file_context"]:
                            st.json(snapshot["file_context"]["files"])
                        else:
                            st.info("No file context available for this state.")

                if "Action" in tabs:
                    with tab_contents[tabs.index("Action")]:
                        action_info = state.action_request.model_dump(exclude_none=True,
                                                                      exclude={"thoughts", "scratch_pad"})
                        st.json(action_info)

                if "Thoughts" in tabs:
                    with tab_contents[tabs.index("Thoughts")]:
                        st.write(thoughts)

                if "Outcome" in tabs:
                    with tab_contents[tabs.index("Outcome")]:
                        outcome_info = state.response.model_dump(exclude_none=True)
                        st.json(outcome_info)

                if "Value Function" in tabs:
                    with tab_contents[tabs.index("Value Function")]:
                        if state.value_function_result:
                            if state.value_function_result.completions:
                                response_content = \
                                    state.value_function_result.completions[0].response['choices'][0]['message'][
                                        'content']
                                st.write(response_content)

                            st.json({
                                "total_reward": state.value_function_result.total_reward,
                                "n_calls": state.value_function_result.n_calls,
                                "average_reward": state.value_function_result.average_reward
                            })

                            st.header("Prompts")
                            for idx, completion in enumerate(state.value_function_result.completions):
                                st.subheader(f"Prompt {idx + 1}")  # sub sub header
                                st.json({
                                    "model": completion.model,
                                    "usage": completion.usage.model_dump() if completion.usage else None
                                })
                                if completion.input:
                                    st.subheader("Input prompts")
                                    for input_idx, input_msg in enumerate(completion.input):
                                        with st.expander(f"Message {input_idx + 1} by {input_msg['role']}",
                                                         expanded=(input_idx == len(completion.input) - 1)):
                                            st.code(input_msg['content'], language="")
                                if completion.response:
                                    st.subheader("Completion response")
                                    st.json(completion.response)
                        else:
                            st.info("No value function result available for this state.")

                if "Completion" in tabs:
                    with tab_contents[tabs.index("Completion")]:
                        if state.completion:
                            st.json({
                                "model": state.completion.model,
                                "usage": state.completion.usage.model_dump() if state.completion.usage else None
                            })

                            if state.completion.input:
                                st.subheader("Inputs")
                                for idx, input_msg in enumerate(state.completion.input):
                                    with st.expander(f"Message {idx + 1} by {input_msg['role']}",
                                                     expanded=(idx == len(state.completion.input) - 1)):
                                        st.json(input_msg)

                            if state.completion.response:
                                st.subheader("Response")
                                st.json(state.completion.response)
                        else:
                            st.info("No completion information available for this state.")

            else:
                st.info("Select a node in the graph or from the dropdown to view details")

        # Move PDF generation and download button outside of columns
        if generate_pdf:
            with st.spinner("Generating PDF..."):
                pdf_content = save_tree_as_pdf(G, pos, focus_node, max_depth, orientation)
            
            st.success("PDF generated successfully!")
            st.download_button(
                label="Download PDF",
                data=pdf_content,
                file_name="trajectory_tree.pdf",
                mime="application/pdf"
            )


def save_tree_as_pdf(G, pos, focus_node=None, max_depth=None, orientation='landscape'):
    # Increase vertical spacing
    for node in pos:
        x, y = pos[node]
        pos[node] = (x, y * 1.1)  # Increase vertical spacing by 50%

    # Function to get subtree
    def get_subtree(G, root, max_depth):
        subtree = nx.DiGraph()
        queue = [(root, 0)]
        while queue:
            node, depth = queue.pop(0)
            if max_depth is not None and depth > max_depth:
                continue
            subtree.add_node(node, **G.nodes[node])
            for child in G.successors(node):
                subtree.add_edge(node, child)
                queue.append((child, depth + 1))
        return subtree

    # Get subtree if focus_node is specified
    if focus_node:
        G = get_subtree(G, focus_node, max_depth)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    # Calculate number of pages needed
    nodes_per_page = 50  # Adjust this value based on your needs
    num_pages = max(1, len(G.nodes) // nodes_per_page)

    # Set figure size based on orientation
    if orientation == 'landscape':
        figsize = (11.69, 8.27)  # A4 landscape
    else:
        figsize = (8.27, 11.69)  # A4 portrait

    # Create a BytesIO object to store the PDF content
    pdf_buffer = BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        for page in range(num_pages):
            fig, ax = plt.subplots(figsize=figsize)

            start_node = page * nodes_per_page
            end_node = min((page + 1) * nodes_per_page, len(G.nodes))

            subgraph = G.subgraph(list(G.nodes)[start_node:end_node])

            # Prepare node colors
            node_colors = []
            for node in subgraph.nodes():
                node_info = G.nodes[node]
                if node_info.get('type') == 'node':
                    reward = node_info.get('value')
                    if reward is not None:
                        # Map reward to a color
                        if reward < 0:
                            color = plt.cm.RdYlGn(0.5 + reward / 200)  # Red to Yellow
                        else:
                            color = plt.cm.RdYlGn(0.5 + reward / 200)  # Yellow to Green
                    else:
                        color = 'gray'
                elif node_info.get('type') == 'action':
                    color = '#8c9aca'  # Grayish blue
                else:
                    color = 'lightblue'
                node_colors.append(color)

            nx.draw(subgraph, pos, ax=ax, with_labels=False, node_size=1000, node_color=node_colors,
                    font_size=8, arrows=True)

            # Add node information as text
            for node in subgraph.nodes():
                x, y = pos[node]
                node_info = G.nodes[node]
                if node_info.get('type') == 'action':
                    info_text = node_info.get('name', 'unknown')
                elif node_info.get('name') == 'Finished':
                    info_text = 'Finished'
                else:
                    info_text = node
                
                ax.text(x, y, info_text, ha='center', va='center', fontsize=6)

            # Create colorbar for rewards
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-100, vmax=100))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, aspect=30, shrink=0.5)
            cbar.set_label('Reward', fontsize=8)
            cbar.ax.tick_params(labelsize=6)

            # Add legend for other node types
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='#8c9aca', label='Action'),
                plt.Rectangle((0, 0), 1, 1, fc='gray', label='No reward'),
            ]
            
            # Place legend below the colorbar
            ax.legend(handles=legend_elements, loc='lower right', fontsize=6, title='Node Types', title_fontsize=8)

            plt.title(f"Trajectory Tree - Page {page + 1}/{num_pages}")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # Return the PDF content as bytes
    return pdf_buffer.getvalue()