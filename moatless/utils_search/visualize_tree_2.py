import logging
import os
import graphviz
from typing import Any, Dict
from pydantic import BaseModel


logger = logging.getLogger(__name__)

class MCTSVisualizer:
    def __init__(self, name='mcts_tree'):
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
        self.all_nodes: Dict[str, 'MCTSNode'] = {}  # Store all nodes
        self.name = name
        # self.excluded_attrs = ['next_completion_messages',
        #                        'last_completion_messages',
        #                        ]
        self.graph.attr('edge', fontname='Arial', fontsize='8', labelDistance='0.3')
        self.included_attrs = ['id', 'parent', 'step', 'reward',
                               'value', 'raw_value']

    def update_graph(self, root: 'MCTSNode'):
        self.add_node_to_graph(root, is_root=True)
        self.save()

    def add_node_to_graph(self, node: 'MCTSNode', is_root: bool = False):
        if node.id not in self.all_nodes:
            self.all_nodes[node.id] = node
            label = self._create_node_label(node)
            fillcolor = self.color_map.get(node.state.name, '#FFFFFF')
            style = "filled"
            self.graph.node(str(node.id), label, shape='box', fillcolor=fillcolor, style=style,
                            color=f"/spectral11/{(len(self.all_nodes) % 11) + 1}")
            
            if node.parent:
                edge_label = self._create_edge_label(node)
                self.graph.edge(str(node.parent.id), str(node.id), label=edge_label)

        for child in node.children:
            self.add_node_to_graph(child)

    def _create_node_label(self, node: 'MCTSNode') -> str:
        label = f"Node{node.id} {node.state.name}\n"

        if node.state.action_request:
            label += "Action:\n"
            label += self._format_dict(node.state.action_request.model_dump(exclude={"thoughts", "scratch_pad"}, exclude_none=True, exclude_unset=True), indent="  ")

        label += f"\n\nVisits: {node.visits}"
        if node.visits:
            label += f"\nValue: {node.value}"
            label += f"\nAvg reward: {node.avg_reward()}"
        
        return label.rstrip()

    def _format_dict(self, d: Dict[str, Any], indent: str = "", max_width: int = 40, max_chars: int | None = None) -> str:
        result = []
        for k, v in d.items():
            formatted_value = self._format_value(v, indent + " ", max_width, max_chars)
            if "\n" in formatted_value:
                result.append(f"{indent}{k}:\n{formatted_value}")
            else:
                result.append(f"{indent}{k}: {formatted_value}")
        return "\n".join(result)

    def _format_value(self, value: Any, indent: str = "", max_width: int = 40, max_chars: int | None = None) -> str:
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_none=True, exclude_unset=True)
        
        if isinstance(value, dict):
            return self._format_dict(value, indent + " ", max_width, max_chars)
        elif isinstance(value, list):
            if all(isinstance(item, (int, float, str, bool)) for item in value):
                return str(value)
            return "\n".join(f"{indent}- {self._format_value(item, indent + ' ', max_width, max_chars)}" for item in value)
        elif isinstance(value, str):
            return self._wrap_text(value, max_width, max_chars)
        return str(value)

    def _create_edge_label(self, node: 'MCTSNode') -> str:
        if node.state.previous_state and node.state.previous_state.outcome:
            label = "Outcome:\n"
            label += self._format_dict(node.state.previous_state.outcome, max_width=40, max_chars=200)
            return label
        return ""

    def _wrap_text(self, text: str, max_width: int, max_chars: int | None = None) -> str:
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

    def save(self):
        self.graph.clear()  # Clear the previous graph
        if not self.all_nodes:
            return
        for node_id, node in self.all_nodes.items():
            label = self._create_node_label(node)
            fillcolor = self.color_map.get(node.state.name, '#FFFFFF')
            style = "filled"
            self.graph.node(str(node_id), label, shape='box', fillcolor=fillcolor, style=style,
                            color=f"/spectral11/{(len(self.all_nodes) % 11) + 1}")
            if node.parent:
                self.graph.edge(str(node.parent.id), str(node_id), label=self._create_edge_label(node))
        try:
            # Get the absolute path of the output file
            output_file = self.graph.render(self.name, format='png', cleanup=True)
            full_path = os.path.abspath(output_file)
            logger.info(f"Graph saved to: {full_path}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            logger.error(f"Attempted to save to: {os.path.abspath(self.name)}.png")