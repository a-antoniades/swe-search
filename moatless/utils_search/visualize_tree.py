import logging
import os
import graphviz
from typing import Any, Dict, List

from moatless.state import AgenticState

logger = logging.getLogger(__name__)

class MCTSVisualizer:
    def __init__(self, name='mcts_tree', max_field_length=250):
        self.graph = graphviz.Digraph(comment='MCTS Tree')
        self.graph.attr(rankdir='TB', size='12,12', dpi='300')
        self.graph.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')
        self.graph.attr('edge', fontname='Arial', fontsize='8')
        self.color_map = {
            'EditCode': '#FFB3BA',  # Light pink
            'PlanToCode': '#BAFFC9',  # Light green
            'IdentifyCode': '#BAE1FF',  # Light blue
            'SearchCode': '#FFFFBA',  # Light yellow
            'Finished': '#E6E6E6',  # Light gray
        }
        self.resolved_color = 'purple'  # Orange for resolved nodes
        self.all_nodes: Dict[str, 'MCTSNode'] = {}  # Store all nodes
        self.name = name
        self.graph.attr('edge', fontname='Arial', fontsize='8', labelDistance='0.3')
        self.included_attrs = ['id', 'duplicate', 'parent', 'step', 'reward',
                               'value', 'raw_value', 'mean_traj_reward',
                               'feedback', 'resolved']
        self.max_field_length = max_field_length
        self.chosen_node: 'MCTSNode' = None  # Store the final chosen node

    def update_graph(self, root: 'MCTSNode'):
        self.add_node_to_graph(root, is_root=True)
        self.save()

    def add_node_to_graph(self, node: 'MCTSNode', is_root: bool = False):
        if node.id not in self.all_nodes:
            self.all_nodes[node.id] = node
            label = self._create_node_label(node)
            fillcolor = self.color_map.get(node.state.name, '#FFFFFF')
            
            # Use color gradients for resolved nodes
            if node.resolved:
                fillcolor = f"{fillcolor}:{self.resolved_color}"
                style = "filled,bold"
            else:
                style = "filled"
            
            # Add a border for duplicate nodes
            if node.duplicate:
                style += ",dashed"
                penwidth = "3"
            else:
                penwidth = "1"
            
            self.graph.node(str(node.id), label, shape='box', fillcolor=fillcolor, style=style,
                            color=f"/spectral11/{(len(self.all_nodes) % 11) + 1}", penwidth=penwidth)
            
            if node.parent:
                edge_label = self._create_edge_label(node)
                edge_color = 'purple' if self._is_in_chosen_path(node) else 'black'
                self.graph.edge(str(node.parent.id), str(node.id), label=edge_label, color=edge_color)

        for child in node.children:
            self.add_node_to_graph(child)

    def _create_node_label(self, node: 'MCTSNode') -> str:
        label = f"Node{node.id}\n{node.state.name}\n"

        state_dump = node.state.model_dump(exclude={'id', 'previous_state', 'next_states', 'origin_state'})
        label += "\n".join([f"{k}: {self._truncate_text(str(v))}" for k, v in state_dump.items()])

        for attr, value in vars(node).items():
            if attr in self.included_attrs and not attr.startswith('__'):
                action_text = self._wrap_text(self._truncate_text(self._format_action(value)), 30)
                label += f"{attr}:\n{action_text}\n"
        return label.rstrip()

    def _create_edge_label(self, node: 'MCTSNode') -> str:
        label = f"Action:\n"
        if node.state.previous_state and isinstance(node.state.previous_state, AgenticState):
            action_request = node.state.previous_state.action_request
            if action_request:
                label += "\n".join([f"  {k}: {self._truncate_text(str(v))}" for k, v in action_request.model_dump(
                    exclude={"thoughts", "scratch_pad"}).items()])
            else:
                label += "No action request"
        else:
            label += "No previous state"
        return self._wrap_text(label, 20)
    
    def _format_action(self, action: Any) -> str:
        if isinstance(action, dict):
            return str({k: v for k, v in action.items() if k != 'thoughts'})
        return str(action)

    def _wrap_text(self, text: str, max_width: int) -> str:
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(' '.join(current_line))
        return '\\n'.join(lines)

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_field_length:
            return text
        return text[:self.max_field_length - 3] + "..."

    def save(self):
        self.graph.clear()  # Clear the previous graph
        if not self.all_nodes:
            return
        for node_id, node in self.all_nodes.items():
            label = self._create_node_label(node)
            fillcolor = self.color_map.get(node.state.name, '#FFFFFF')
            
            # Use color gradients for resolved nodes
            if node.resolved:
                fillcolor = f"{fillcolor}:{self.resolved_color}"
                style = "filled,bold"
            else:
                style = "filled"
            
            self.graph.node(str(node_id), label, shape='box', fillcolor=fillcolor, style=style,
                            color=f"/spectral11/{(len(self.all_nodes) % 11) + 1}")
            if node.parent:
                edge_color = 'purple' if self._is_in_chosen_path(node) else 'black'
                self.graph.edge(str(node.parent.id), str(node_id), label=self._create_edge_label(node), color=edge_color)
        try:
            # Get the absolute path of the output file
            output_file = self.graph.render(self.name, format='png', cleanup=True)
            full_path = os.path.abspath(output_file)
            logger.info(f"Graph saved to: {full_path}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            logger.error(f"Attempted to save to: {os.path.abspath(self.name)}.png")

    def highlight_chosen_path(self, final_node: 'MCTSNode'):
        """
        Updates the graph to highlight the chosen path with purple edges.

        Args:
        final_node ('MCTSNode'): The final node in the chosen path.
        """
        self.chosen_node = final_node
        self.save()

    def _is_in_chosen_path(self, node: 'MCTSNode') -> bool:
        """
        Checks if the node is part of the chosen path.

        Args:
        node ('MCTSNode'): The node to check.

        Returns:
        bool: True if the node is part of the chosen path, False otherwise.
        """
        if not self.chosen_node:
            return False
        current = self.chosen_node
        while current:
            if current.id == node.id:
                return True
            current = current.parent
        return False