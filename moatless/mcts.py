import os
from typing import Tuple, Optional, Dict, Any, List
from collections import OrderedDict
import traceback
import logging
import random
import math
import numpy as np
import json
from copy import deepcopy
import time
import cProfile
import pstats
import io
from functools import wraps

from .edit import PlanToCode, EditCode
from .find import IdentifyCode
from .repository import GitRepository
from .schema import TestStatus
from .settings import Settings
from .transition_rules import TreeSearchSettings
from .utils_search.visualize_tree import MCTSVisualizer
from .search.reward import LLM_Value_Function

from moatless.state import (
    AgenticState,
    Finished,
    Pending,
    Rejected, Visit, State,
)
from moatless.state import ActionRequest
from moatless.utils_search.misc import save_to_json
from moatless.search.reward import MessageCreator

logger = logging.getLogger('mcts')

tree_logger = logging.getLogger('mcts_tree')

class MCTSNode:
    def __init__(self,
                 id,
                 state: AgenticState,
                 parent=None,
                 origin_state: Optional[AgenticState] = None,
                 last_action=None,
                 last_completion_messages=None,
                 last_completion_response=None,
                 next_completion_messages=None,
                 file_context=None,
                 loop=None,
                 duplicate=False,
                 step=0, **kwargs):

        self.id = id
        self.state = state
        self.origin_state = origin_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.raw_value = 0
        self.duplicate = duplicate
        self.last_action = last_action
        self.last_completion_messages = last_completion_messages
        self.last_completion_response = last_completion_response
        self.next_completion_messages = next_completion_messages
        self.file_context = file_context
        self.loop = loop
        self.step = step
        self.trajectory = []
        self.resolved = False

        self.exploitation = 0
        self.exploration = 0
        self.depth_penalty = 0

        self.resolved = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    def expanded_count(self) -> int:
        return len(self.children)
    
    def has_children(self) -> bool:
        return len(self.children) > 0

    def avg_reward(self):
        return self.value / self.visits if self.visits > 0 else 0

    def get_depth(self) -> int:
        depth = 0
        node = self
        while node.parent:
            depth += 1
            node = node.parent
        return depth

    def get_mean_traj_reward(self) -> float:
        """
        Calculate the mean trajectory reward for this node.

        Returns:
            float: The mean trajectory reward.
        """
        trajectory_rewards = []
        node = self
        while node is not None:
            trajectory_rewards.append(node.value / node.visits if node.visits > 0 else 0)
            node = node.parent

        return sum(trajectory_rewards) / len(trajectory_rewards) if trajectory_rewards else 0


class MCTS:

    def __init__(self, loop,
                 c_param: float = 1.41,
                 max_expansions: int = 2,
                 min_finished_transitions_for_early_stop: int | None = None,
                 max_completed_trajectories: int | None = None,
                 reward_threshold: float | None = None,
                 states_to_explore: List[str] | None = None,
                 provide_feedback: bool = False,
                 debate: bool = False,
                 value_function_model: str | None = None,
                 best_first: bool = True,
                 settings: TreeSearchSettings | None = None,
                 enable_vizualizer: bool = True,
                 **kwargs
        ):
        """
        Initialize the Monte Carlo Tree Search (MCTS) algorithm.

        Args:
            loop: The main execution loop object.
            c_param (float): Exploration parameter for uct1 formula. Default is 1.41.
            max_expansions (int): Maximum number of expansions (children) per node. Default is 2.
            min_finished_transitions_for_early_stop (int): Minimum number of completed trajectories before complete. Default is None
            reward_threshold(float): Reward threshold to stop search early. Default is None.
            provide_feedback (bool): If True, provide feedback to previously visited nodes. Default is True.
        """
        self.node_count = 1  # Class variable to keep track of total nodes across all searches
        self.loop = loop
        self.taskname = self.loop._trajectory_path.replace('trajs', 'flow_chart').replace('.json', '')
        self.message_creator = MessageCreator()
        self.visualizer = MCTSVisualizer(name=self.taskname)
        self.root = MCTSNode(id=self.loop.state.id,
                             state=self.loop.state,
                             loop=loop)
        self.nodes = OrderedDict({self.root.id: self.root})
        self.c_param = c_param
        self.enable_vizualizer = enable_vizualizer

        if settings:
            self.provide_feedback = settings.provide_feedback
            self.debate = settings.debate
            self.max_expansions = settings.max_expansions
            self.max_iterations = settings.max_iterations
            self.reward_threshold = settings.reward_threshold
            self.min_finished_transitions_for_early_stop = settings.min_finished_transitions_for_early_stop
            self.max_completed_trajectories = settings.max_finished_transitions
            self.states_to_explore = settings.states_to_explore
            self.value_function_model = settings.value_function_model or Settings.cheap_model
            self.best_first = settings.best_first
            self.value_function_temperature = settings.value_function_model_temperature
        else:
            self.provide_feedback = provide_feedback
            self.debate = debate
            self.max_iterations = self.loop._max_transitions
            self.max_expansions = max_expansions
            self.reward_threshold = reward_threshold
            self.min_finished_transitions_for_early_stop = min_finished_transitions_for_early_stop
            self.max_completed_trajectories = max_completed_trajectories
            self.states_to_explore = states_to_explore or ["SearchCode", "PlanToCode"]
            self.value_function_model = value_function_model or kwargs.get("model")
            self.best_first = best_first
            self.value_function_temperature = kwargs.get("temperature") or 0.0

        self.provide_feedback_thresh = 101
        self.max_depth = 20
        self.context_history = {}
        self.first_pass_done = False

        def create_path(new_dir, extension=None, filename=None):
            base_path, original_filename = os.path.split(self.loop._trajectory_path)
            instance_name = os.path.basename(base_path)
            logger.info(f"base_path: {base_path}, original_filename: {original_filename}, new_dir: {new_dir},instance_name: {instance_name}")
            if filename:
                original_filename = filename
            if extension:
                original_filename = original_filename.replace('json', extension)
            return os.path.join(os.path.dirname(base_path), instance_name,
                                new_dir, original_filename)

        # Create paths for different file types
        self.value_fun_taskname = create_path('search', filename='rewards.json')
        self.tree_filename = create_path('search')
        self.solutions_filename = create_path('search', filename='solutions.json')
        self.ascii_tree_filename = create_path('search', filename='ascii_tree.txt')
        self.nodes_filename = create_path('search', filename='nodes.json')

        logger.info(f"Self.value_fun_taskname: {self.value_fun_taskname}")
        self.value_function = LLM_Value_Function(filename=self.value_fun_taskname,
                                                 model=self.value_function_model,
                                                 temperature=self.value_function_temperature)

    def save_search_tree(self):
        os.makedirs(os.path.dirname(self.tree_filename), exist_ok=True)

        traj_vars_ommit = ["name", "initial_message"]

        def save_trajectory(trajectory):
            if isinstance(trajectory, dict):
                return {k: v for k, v in trajectory.items() if k not in traj_vars_ommit}
            else:
                return trajectory

        def build_tree_dict(node):
            node_dict = {
                'node_info': {
                    'id': node.id,
                    'visits': node.visits,
                    'value': node.value,
                },
                'trajectory': save_trajectory(node.trajectory),
                'children': {}
            }
            for child in node.children:
                node_dict['children'][child.id] = build_tree_dict(child)
            return node_dict

        tree_dict = build_tree_dict(self.root)
        full_dict = {
            "Reported Issue": self.loop._trajectory._initial_message,
            "Search Tree": tree_dict,
        }

        # Save the tree dictionary to a JSON file
        save_to_json(full_dict, self.tree_filename)

        logger.info(f"Hierarchical search tree saved to {self.tree_filename}")

        return full_dict

    def filter_nodes(self, nodes: List[MCTSNode]) -> List[MCTSNode]:
        nodes_to_explore = []
        states_to_explore = ["PlanToCode"]

        for node_id, node in nodes.items():
            if isinstance(node.state, (Finished, Rejected)):
                continue
            elif len(node.children) >= self.max_expansions:
                logger.info(f"Node {node.id} has {len(node.children)} children, skipping")
                continue
            elif node.state.name in states_to_explore:
                logger.info(f"Node {node.id} has state {node.state.name}, adding to explore")
                nodes_to_explore.append(node_id)
        return nodes_to_explore

    def get_best_explore_from_uct(self, parent: MCTSNode, nodes: List[MCTSNode]) -> MCTSNode:
        return max(nodes, key=lambda n: self.uct_score(parent, n))

    def filter_mature_nodes(self) -> List[MCTSNode]:
        nodes_to_explore = []
        ignored_nodes: Dict[str, List[int]] = {
            "terminal": [],
            "max_expanded": [],
            "low_reward": [],
            "finished": [],
            "duplicates": []
        }

        for node_id, node in self.nodes.items():
            if not node.parent:
                # Return the root node in the first iteration, otherwise skip
                if node.children:
                    continue
                else:
                    return [node]

            if isinstance(node.state, (Finished, Rejected)):
                ignored_nodes["terminal"].append(node_id)
                continue

            if node.duplicate:
                ignored_nodes["duplicates"].append(node_id)
                continue

            # # if we're at the beginning, expand more nodes
            # if node.get_depth() <= 1:
            #     if len(node.children) <= 5:
            #         nodes_to_explore.append(node)
            #         continue

            if node.expanded_count() >= self.max_expansions: # or node.expanded_count() >= node.state.max_expansions:
                ignored_nodes["max_expanded"].append(node_id)
                continue

            # don't continue to explore nodes that has a reward higher than 'reward_threshold'
            try:
                if self.reward_threshold and any(child.raw_value and child.raw_value >= self.reward_threshold for child in node.children):
                    ignored_nodes["finished"].append(node_id)
                    continue

            except Exception as e:
                logger.info(f"Error in finished nodes for Node{node.id}")
                logger.info(f"Node children: {node.children}")
                logger.info(f"children raw_value: {[child.raw_value for child in node.children]}")
                logger.info(f"reward_threshold: {self.reward_threshold}")
                raise e

            n_finished_states = sum(1 for child in node.children if isinstance(child.state, Finished))
            if n_finished_states > 2:
                ignored_nodes["finished"].append(node_id)
                continue

            # # Ignore nodes with a reward < 0
            # if node.raw_value < 0:
            #     ignored_nodes["low_reward"].append(node_id)
            #     continue

            # Calculate average reward for the node and its best child
            node_avg_reward = node.value / node.visits if node.visits > 0 else 0
            best_child_avg_reward = max((child.value / child.visits for child in node.children if child.visits > 0),
                                        default=0)

            # If the node's average reward is less than its best child's, it might still be worth exploring
            if node_avg_reward >= best_child_avg_reward:
                ignored_nodes["low_reward"].append(node_id)

            if node.state.name not in self.states_to_explore and node.expanded_count() > 0:
                continue

            nodes_to_explore.append(node)

        # # If there are no nodes to explore, return the initial search (1) node and top 5 nodes with highest raw_value
        # if len(nodes_to_explore) == 0:
        #     self.first_pass_done = True
        #     top_nodes = sorted(
        #         [node for node in self.nodes.values() if node.expanded_count() < 5],
        #         key=lambda x: x.raw_value,
        #         reverse=True
        #     )[:5]
        #     search_node = self.nodes[1]
        #     nodes_to_explore = [search_node] + top_nodes

        summary = f"Total nodes ignored: {sum(len(nodes) for nodes in ignored_nodes.values())}"
        for category, nodes in ignored_nodes.items():
            summary += f"\n - {category.capitalize()} nodes ignored: {len(nodes)}"
            if nodes:
                summary += f" ({', '.join(map(str, nodes))})"
        summary += f"\nNodes available for expansion: {' '.join([f'({str(node.state.name)}, {str(node.id)})' for node in nodes_to_explore])}"
        if nodes_to_explore:
            summary += f" ({', '.join(str(node.id) for node in nodes_to_explore)})"
        logger.info(summary)

        return nodes_to_explore

    def uct_score(self, parent: MCTSNode, node: MCTSNode,
                exploration_weight: float = 1.0, depth_weight: float = 0.8,
                depth_bonus_factor: float = 200.0,
                high_value_threshold: float = 55.0,
                low_value_threshold: float = 50.0,
                very_high_value_threshold: float = 75.0,
                high_value_leaf_bonus_constant: float = 20.0,
                high_value_bad_children_bonus_constant: float = 20.0,
                high_value_child_penalty_constant: float = 5.0) -> float:
        """Compute the UCT score with additional bonuses and penalties based on node characteristics."""
        if node.visits == 0:
            return float('inf')

        # Calculate exploitation and exploration components
        # exploitation = node.raw_value
        exploitation = node.raw_value
        exploration = exploration_weight * math.sqrt(math.log(parent.visits) / node.visits)

        depth = node.get_depth()

        # Depth-based exploration bonus
        if not self.first_pass_done:
            if depth <= 1 and node.expanded_count(): # not used for now
                depth_bonus = depth_bonus_factor * np.exp(-depth_weight * (depth - 1))
            else:
                depth_bonus = 0

        # Depth penalty for very deep nodes
        depth_penalty = depth_weight * math.sqrt(depth)

        # Initialize bonuses and penalties
        high_value_leaf_bonus = 0.0
        high_value_bad_children_bonus = 0.0
        high_value_child_penalty = 0.0
        high_value_parent_bonus = 0.0

        # Additional bonus for not expanded nodes with high reward
        if not node.children and exploitation >= high_value_threshold:
            high_value_leaf_bonus = high_value_leaf_bonus_constant

        # Additional bonus for nodes with high reward that expanded to low-reward nodes
        child_values = [child.raw_value for child in node.children]
        if node.children and exploitation >= high_value_threshold:
            child_values = [child.raw_value for child in node.children]

            if len(child_values) < 2:
                avg_child_value = sum(child_values) / len(child_values)
                if avg_child_value <= low_value_threshold:
                    high_value_bad_children_bonus = (exploitation - avg_child_value) * 5

            # Penalty for nodes with a child with very high reward
            max_child_value = max(child_values)
            if max_child_value >= very_high_value_threshold:
                high_value_child_penalty = high_value_child_penalty_constant
        
        # additional bonus for nodes with low reward that haven't been expanded yet but have high reward parents
        if node.parent and not node.has_children(): 
            if node.parent.raw_value > high_value_threshold:
                if exploitation <= low_value_threshold:
                    high_value_parent_bonus = (high_value_threshold - exploitation) * 5

        # Store components for debugging or analysis
        node.exploitation = exploitation
        node.exploration = exploration + depth_bonus
        node.depth_penalty = depth_penalty
        node.high_value_leaf_bonus = high_value_leaf_bonus
        node.high_value_bad_children_bonus = high_value_bad_children_bonus
        node.high_value_child_penalty = high_value_child_penalty
        node.high_value_parent_bonus = high_value_parent_bonus

        # Compute the final UCT score
        return (
            exploitation +
            depth_bonus -
            depth_penalty +
            high_value_leaf_bonus +
            high_value_bad_children_bonus -
            high_value_child_penalty +
            high_value_parent_bonus
        )

    def select(self, best_first:  bool = True) -> MCTSNode | None:
        # Get all unexpanded or promising nodes
        available_nodes = self.filter_mature_nodes()

        if not available_nodes:
            logger.info("No available nodes to expand.")
            return None

        try:
            if best_first:
                selected_node = max(
                    available_nodes,
                    key=lambda n: self.uct_score(
                        n.parent, n,
                    )
                )
            else:
                # Calculate UCT scores for available nodes
                uct_scores = [
                    self.uct_score(
                        n.parent, n,
                    ) if n.parent else 0 for n in available_nodes
                ]

                # Apply softmax to UCT scores to get probabilities
                softmax_scores = np.exp(uct_scores - np.max(uct_scores))  # Subtract max for numerical stability
                weights = softmax_scores / softmax_scores.sum()

                # Select a node using weighted random sampling
                selected_node = random.choices(available_nodes, weights=weights, k=1)[0]
        
        except ValueError as e:
            logger.error(f"Error in node selection: {e}")
            logger.info(f"UCT scores: {uct_scores}")
            logger.info(f"Softmax scores: {softmax_scores}")
            logger.info(f"Weights: {weights}")
            # Fallback to uniform selection if there's an issue
            selected_node = random.choice(available_nodes)

        logger.info(
            f"Node{selected_node.id} with state {selected_node.state.name} {selected_node.state.id} selected for expansion.")
        return selected_node

    def update_visits_for_state(self, target_state: AgenticState, visit_increment: int = 1):
        for node in self.nodes.values():
            if node.state == target_state:
                node.visits += visit_increment
                logger.info(f"Updated Node{node.id} visits to {node.visits}")

    def run_search(self) -> AgenticState | None:

        # Execute Pending state
        new_state = self.loop._execute_state_until_transition()
        self.node_count += 1
        node = MCTSNode(
            id=new_state.id,
            state=new_state,
            parent=self.root,
            loop=self.loop,
            step=1,
        )
        self.nodes[self.root.id].children.append(node)
        self.nodes[node.id] = node

        reward = 0

        for i in range(self.max_iterations):
            total_cost = self.loop.total_cost()
            if total_cost > self.loop._max_cost:
                logger.warning(
                    f"Max cost reached ({total_cost} > {self.loop._max_cost}). Exiting."
                )
                break

            # if isinstance(node.state, AgenticState) and reward >= 75 and not node.duplicate:
            #     logger.info(
            #         f"Continue to expand Node{node.id} with reward {reward} that is 75 or higher")
            # else:
            node = self.select(best_first=self.best_first)

            if node is None or isinstance(node.state, (Finished, Rejected)):
                logger.info("No node selected for expansion, stopping search")
                break  # No more nodes to explore or we've reached a terminal state

            logger.info(
                f"Starting iteration {i + 1}/{self.max_iterations} with Node{node.id} {node.state.trace_name}")

            child = self.expand(node)
            if child is None:
                logger.error(f"Error expanding Node{node.id}, skipping")
                continue

            node = child
            if node.duplicate:
                log_tree = self.generate_ascii_tree(self.root, child)
                tree_logger.info(f"Expanded Node{node.id} with duplicated state.\nCurrent tree:\n{log_tree}")
                continue

            if isinstance(node.state, Rejected):
                log_tree = self.generate_ascii_tree(self.root, node)
                tree_logger.info(f"Rejected Node{node.id}.\nCurrent tree:\n{log_tree}")
                continue

            reward, explanation = self.simulate(node)

            self.backpropagate(node, reward, explanation)

            if self.enable_vizualizer:
                self.visualizer.update_graph(self.root)

            # Run evaluation on every finished trajectory
            if isinstance(node.state, Finished) or isinstance(node.state, Rejected):
                self.evaluate(node)
                best_finish_node, best_mean_reward = self.get_best_trajectory()

            try:
                log_tree = self.generate_ascii_tree(self.root, node)
                self.save_ascii_tree(log_tree, self.ascii_tree_filename)
                tree_logger.info(f"Expanded Node{node.id} with reward {reward}.\nCurrent tree:\n{log_tree}")
            except Exception as e:
                logger.error(f"Error in saving ascii tree for Node{node.id}: {e}")

            self.loop._trajectory._maybe_persist()

            # Export nodes information after each iteration
            self.export_nodes(self.nodes_filename)

            if isinstance(node.state, Finished):
                finished_nodes = [node for node in self.nodes.values() if
                                  isinstance(node.state, Finished) and not node.duplicate]
                best_finish_node, best_mean_reward = self.get_best_trajectory()
                if self.enable_vizualizer:
                    self.visualizer.highlight_chosen_path(best_finish_node)

                filtered_nodes = []
                patches = set()
                for node in finished_nodes:
                    if node.state.output:
                        diff = node.state.output["diff"]
                        if diff not in patches:
                            patches.add(diff)
                            filtered_nodes.append(node)

                if len(filtered_nodes) < len(finished_nodes):
                    logger.info(f"Filtered {len(finished_nodes) - len(filtered_nodes)} duplicate paths from {len(finished_nodes)} finished paths.")
                finished_nodes = filtered_nodes

                if self.min_finished_transitions_for_early_stop and len(finished_nodes) >= self.min_finished_transitions_for_early_stop and best_finish_node.raw_value >= self.reward_threshold:
                    logger.info(
                        f"Best path with reward {best_finish_node.raw_value} (mean reward {best_mean_reward}) if higher than {self.reward_threshold}, stopping search")
                    break

                if self.max_completed_trajectories and len(finished_nodes) >= self.max_completed_trajectories:
                    logger.info(
                        f"Max completed trajectories reached ({len(finished_nodes)} >= {self.max_completed_trajectories}), stopping search")
                    break

        best_node, reward = self.get_best_trajectory()
        if self.enable_vizualizer:
            self.visualizer.highlight_chosen_path(best_node)
            self.visualizer.update_graph(self.root)  # Final update after all iterations

        if best_node:
            logger.info(f"Return best Node{best_node.id} with total reward: {reward}")
            return best_node.state
        else:
            logger.warning("No best node found, returning None")
            return None

    def get_best_trajectory(self) -> Tuple[MCTSNode, int]:
        def save_diff(node_id, diff, filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            try:
                with open(filename, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}

                    # Convert node_id to string to ensure consistent key type
                    node_id = str(node_id)

                    # Update the data with the new diff, overwriting if key already exists
                    data[node_id] = diff

                    # Rewrite the entire file with updated data
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                logger.info(f"Saving diff for Node{node_id} at {filename}")
            except FileNotFoundError:
                with open(filename, 'w') as f:
                    json.dump({str(node_id): diff}, f, indent=2)

        finished_nodes = [node for node in self.nodes.values() if isinstance(node.state, Finished) and not node.duplicate]
        logger.info(f"Finished nodes ({len(finished_nodes)}): {' '.join(str(node.id) for node in finished_nodes)}")

        if not finished_nodes:
            logger.warning("No finished nodes found. Returning the best path based on uct scores.")
            return self.get_best_path_uct()

        best_finish_node = None
        best_mean_reward = float('-inf')
        trajectories_mean_rewards = []

        for finished_node in finished_nodes:
            node = finished_node
            if finished_node.state is not None:
                if hasattr(finished_node.state, "output"):
                    if finished_node.state.output:
                        diff = finished_node.state.output["diff"]
                        save_diff(finished_node.id, diff, self.solutions_filename)
                        logger.info(f"saving diff for Node{finished_node.id} at {self.solutions_filename}")
                else:
                    logger.info(f"No diff found for Node{finished_node} {finished_node.id}")
            else:
                logger.info(f"finished_node.state is None for Node{finished_node.id}")

            # mean_reward = finished_node.raw_value
            mean_reward = finished_node.get_mean_traj_reward()

            self.nodes[finished_node.id].mean_traj_reward = mean_reward
            trajectories_mean_rewards.append(mean_reward)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_finish_node = finished_node

        logger.info(f"Mean Rewards for finished trajectories: {trajectories_mean_rewards}")

        if best_finish_node:
            logger.info(f"Best finished path finished on Node{best_finish_node.id} with mean reward: {best_mean_reward}")
            return best_finish_node, best_mean_reward
        else:
            logger.info("No valid finished path found. This should not happen if there are finished nodes.")
            return None, 0

    def get_best_path_uct(self) -> Tuple[MCTSNode, int]:
        trajectory = []
        node = self.root
        total_reward = 0

        while node.children:
            best_child = max(node.children, key=lambda c: self.uct_score(node, c, exploration_weight=0))
            if best_child.last_action:
                trajectory.append((best_child.last_action, best_child.state))
            total_reward += best_child.value / best_child.visits if best_child.visits > 0 else 0
            node = best_child

        return node, total_reward

    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        try:
            state = self.loop.revert_to_state(node.state)
        except Exception as e:
            logger.exception(f"Error in expanding Node{node.id}")
            return None

        logger.info(f"Expanding Node{node.id} {self.loop.state.name} origin id {self.loop.state.id}")
        state_name = f"{state.name} ({state.id})"
        feedback = None

        if node.expanded_count():
            state = self.loop.clone_current_state()

            if state.provide_feedback:
                # If the node already been expanded feedback is generated

                visited_children = [child for child in node.children if child.visits > 0]
                if not visited_children or not self.provide_feedback:
                    feedback = None
                elif visited_children:
                    best_child = min(visited_children, key=lambda n: n.value / n.visits if n.visits > 0 else 0)
                    if best_child.raw_value < self.provide_feedback_thresh:
                        feedback = self.message_creator.generate_feedback(best_child)
                        logger.info(
                                f"Node{node.id} {self.loop.state.name} has a raw value of {node.raw_value}, providing feedback")
                    else:
                        feedback = None

                # if raw_value is < self.provide_feedback_thresh, provide feedback
                # if node.raw_value <= self.provide_feedback_thresh:
                state.feedback = feedback
                logger.info(
                    f"Node{node.id} has been expanded {node.expanded_count()} times. Will use state {state.name} ({state.id}) cloned from {state_name}")

        if self.states_to_explore:
            new_state = self.loop._execute_state_until_transition_to(self.states_to_explore)
        else:
            new_state = self.loop._execute_state_until_transition()

        saved_trajectory = self.loop._trajectory.to_dict()

        step_count = node.step + 1
        self.node_count += 1  # Increment the class variable

        if isinstance(new_state, AgenticState):
            new_state.init()

        child_node = MCTSNode(
            id=len(self.nodes) + 1,
            state=new_state,
            parent=node,
            loop=self.loop,
            step=step_count,
            trajectory=saved_trajectory,
            file_context=new_state.file_context.clone(),
            feedback=feedback
        )

        if self.nodes[node.id].children and self.is_duplicate(node, new_state):
            child_node.duplicate = True

        self.nodes[node.id].children.append(child_node)
        self.nodes[child_node.id] = child_node

        if self.enable_vizualizer:
            self.visualizer.add_node_to_graph(child_node)
        return child_node

    def update_root(self, new_state: AgenticState):
        # Try to find a child node that matches the new state
        for child in self.root.children:
            if child.state == new_state:
                self.root = child
                self.root.parent = None  # The new root has no parent
                logger.info(f"MCTS root updated to existing node: {self.root.id}")
                return

        # If no matching child is found, create a new root
        self.node_count += 1
        self.root = MCTSNode(id=self.node_count, state=new_state)
        logger.info(f"MCTS root reset to new node: {self.root.id}")

    def simulate(self, node: MCTSNode) -> Tuple[float, str]:
        # reward = self.calculate_reward(node.state, node.last_action, node.state)
        # reward = 1

        # LLM reward
        reward, explanation = self.value_function.get_reward(
            node=node,
            debate=self.debate
        )
        return reward, explanation


    def calculate_reward(self, current_state: AgenticState, action: ActionRequest, next_state: AgenticState) -> float:
        # Implement a more comprehensive reward calculation here
        # This should take into account the desirability of the transition
        # For example:
        if isinstance(next_state, Finished):
            return 1.0  # High reward for reaching a finished state
        elif isinstance(next_state, Rejected):
            return -1.0  # Negative reward for reaching a rejected state
        else:
            # Calculate a reward based on how "good" the transition is
            # This could involve comparing some properties of current_state and next_state
            # For now, we'll return a small positive reward for non-terminal transitions
            return 0.1

    def backpropagate(self, node: MCTSNode, result: float, explanation: str):
        path = []

        current_node = node
        current_node.raw_value = result

        source_state_id = current_node.state.id
        # TODO: target_state_id defines the state to which the explanation should be added. It's now set to the new state.
        target_state_id = current_node.state.id

        while current_node is not None:
            if target_state_id == current_node.state.id:
                logger.debug(f"Adding visit with explanation to Node{current_node.id}: {result:.2f} ({explanation})")
                current_node.state.visits.append(Visit(
                    source_state_id=source_state_id,
                    value=result,
                    explanation=explanation
                ))
            else:
                current_node.state.visits.append(Visit(
                    source_state_id=source_state_id,
                    value=result,
                ))

            # Update the current current_node
            current_node.visits = len(current_node.state.visits)
            current_node.value = sum(visit.value for visit in current_node.state.visits)

            path.append(f"Node{current_node.id}_({current_node.visits}, {current_node.value:.2f})")

            current_node = current_node.parent

        logger.info(f"Backpropagation path: {' -> '.join(reversed(path))}")
        if self.enable_vizualizer:
            self.visualizer.update_graph(self.root)

    def best_child(self, node: MCTSNode) -> MCTSNode:
        best = max(node.children, key=lambda n: n.value / n.visits + self.c_param * (2 * node.visits / n.visits) ** 0.5)
        logger.info(f"Best child of Node{node.id}: Node{best.id}")
        return best

    def progressive_widening_probability(self, node: MCTSNode) -> float:
        # prob = self.max_expansions / (node.visits + self.max_expansions)
        # logger.info(f"Node{node.id} - Progressive widening probability: {prob:.2f}")
        prob = 0.2
        return prob

    def generate_ascii_tree(self, root: MCTSNode, current: MCTSNode) -> str:
        tree_lines = ["MCTS Tree"]
        self._append_ascii_node(root, "", True, tree_lines, current)
        return "\n".join(tree_lines)

    def _append_ascii_action(self, node: MCTSNode, prefix: str, is_last: bool, tree_lines: list[str],
                             current: MCTSNode):
        executed_state = node.state.previous_state
        if not isinstance(executed_state, AgenticState) or not executed_state.action_request:
            logger.info(f"Node{node.id} has no action request")
            return

        results = []
        support_state = ""
        if executed_state.name not in ["PlanToCode", "SearchCode"]:
            if executed_state.name == "EditCode":
                support_state += f" -> Edit(lines={executed_state.start_line}-{executed_state.end_line}"

                if len(executed_state.actions):
                    support_state += f", {len(executed_state.actions)} retries"
                support_state += ")"

            if executed_state.name == "IdentifyCode":
                support_state += " -> Identify("
                if executed_state.ranked_spans:
                    support_state += f"search_results={len(executed_state.ranked_spans)}"
                support_state += ")"

            if executed_state.completion and executed_state.completion.usage:
                support_state += f"[in: {executed_state.completion.usage.prompt_tokens}, out: {executed_state.completion.usage.completion_tokens}]"

            if executed_state.outcome:
                if executed_state.outcome.get("updated_file_context", []):
                    file_strs = []
                    for file_with_spans in executed_state.outcome.get("updated_file_context", []):
                        file_name = file_with_spans.file_path.split("/")[-1]
                        file_strs.append(f"{file_name}: {len(file_with_spans.span_ids)}")
                    results.append(f"spans=[{','.join(file_strs)}]")

                if executed_state.outcome.get("message"):
                    results.append(f"message='{executed_state.outcome.get('message').strip()[:20]}...'")

                if executed_state.outcome.get("diff"):
                    diff_lines = executed_state.outcome.get("diff").split("\n")
                    plus_lines = [line for line in diff_lines if line.startswith("+")]
                    minus_lines = [line for line in diff_lines if line.startswith("-")]
                    results.append(f"diff +{len(plus_lines)}, -{len(minus_lines)}")

            executed_state = executed_state.previous_state

        first_state = f"{executed_state.name}{executed_state.id}.{executed_state.action_request.log_name}"
        if executed_state.completion and executed_state.completion.usage:
            first_state += f"[in: {executed_state.completion.usage.prompt_tokens}, out: {executed_state.completion.usage.completion_tokens}]"

        action_log = f"{first_state}{support_state} -> ({', '.join(results)})"

        if node.id == current.id:
            action_log = color_white(action_log)

        tree_lines.append(f"{prefix}└── {action_log}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        self._append_ascii_node(node, child_prefix, True, tree_lines, current)

    def _append_ascii_node(self, node: 'MCTSNode', prefix: str, is_last: bool, tree_lines: list[str],
                           current: MCTSNode):
        state_params = []

        if hasattr(node.state, "diff") and node.state.diff:
            diff_lines = node.state.diff.split("\n")
            plus_lines = [line for line in diff_lines if line.startswith("+")]
            minus_lines = [line for line in diff_lines if line.startswith("-")]
            state_params.append(f"diff +{len(plus_lines)}, -{len(minus_lines)}")

        if hasattr(node.state, "test_results") and node.state.test_results:
            failed = sum(1 for result in node.state.test_results if result.status == TestStatus.FAILED)
            errored = sum(1 for result in node.state.test_results if result.status == TestStatus.ERROR)

            if not failed and not errored:
                result = f"{len(node.state.test_results)} tests passed"
            elif errored == len(node.state.test_results):
                result = f"{len(node.state.test_results)} errors"
            else:
                result = f"{len(node.state.test_results)} tests, {errored} errors, {failed} failures"

            state_params.append(result)

        if node.state.name == "Finished":
            if hasattr(node.state, "output") and node.state.output:
                if node.state.output.get("diff"):
                    diff_lines = node.state.output.get("diff").split("\n")
                    plus_lines = [line for line in diff_lines if line.startswith("+")]
                    minus_lines = [line for line in diff_lines if line.startswith("-")]
                    state_params.append(f"diff +{len(plus_lines)}, -{len(minus_lines)}")

                if node.state.output.get("evaluation_result", None):
                    if node.state.output.get("resolved"):
                        state_params.append(color_green("Resolved"))
                    else:
                        state_params.append(color_red("Failed"))

                    tests_status = node.state.output["evaluation_result"].get("tests_status", {})
                    f2p_fails = len(tests_status.get("fail_to_pass", {}).get("failure", []))
                    p2p_fails = len(tests_status.get("pass_to_pass", {}).get("failure", []))
                    if f2p_fails:
                        state_params.append(color_red(f"f2p_fails={f2p_fails}"))

                    if p2p_fails:
                        state_params.append(color_red(f"p2p_fails={p2p_fails}"))

        elif node.state.name == "Rejected":
            if hasattr(node.state, "output") and node.state.output:
                if node.state.output.get("message"):
                    state_params.append(f"msg='{node.state.output.get('message').strip()[:30]}...'")

                if node.state.output.get("error"):
                    state_params.append(f"error='{node.state.output.get('error').strip()[:30]}...'")

        elif node.file_context:
            state_params.append(f"files={len(node.file_context.files)}")
            state_params.append(f"tokens={node.file_context.context_size()}")

        state_info = f"{node.state.name}{node.state.id}"
        if state_params:
            state_info += f"({', '.join(state_params)})"
        else:
            state_info += f"()"

        if node.id == current.id:
            state_info = color_white(state_info)

        if not node.raw_value:
            reward_str = "0"
        elif node.raw_value >= 75:
            reward_str = color_green(node.raw_value)
        elif node.raw_value <= 0:
            reward_str = color_red(node.raw_value)
        else:
            reward_str = color_yellow(node.raw_value)

        avg_reward = node.get_mean_traj_reward()
        if avg_reward >= 75:
            avg_reward_str = color_green(f"{avg_reward:.1f}")
            node_str = color_green(f"Node{node.id} [{avg_reward_str}/{reward_str}]")
        elif avg_reward < 0:
            avg_reward_str = color_red(f"{avg_reward:.1f}")
            node_str = color_red(f"Node{node.id} [{avg_reward_str}/{reward_str}]")
        else:
            avg_reward_str = color_yellow(f"{avg_reward:.1f}")
            node_str = color_yellow(f"Node{node.id} [{avg_reward_str}/{reward_str}]")

        if node.duplicate:
            tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}Node{node.id} {state_info} (duplicate)")
        elif node.state.name == "Rejected":
            node_name = color_red(f"Node{node.id}")
            tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{node_name} {state_info}")
        else:
            tree_lines.append(
                f"{prefix}{'└── ' if is_last else '├── '}{node_str} {state_info} (expansions: {node.expanded_count()}, avg_reward: {avg_reward_str}, reward: {reward_str}, visits: {node.visits}, uct_score: {self.uct_score(node.parent, node) if node.parent else 0:.2f}), exploit: {node.exploitation:.2f}, explore: {node.exploration:.2f}, depth_penalty: {node.depth_penalty:.2f}")

            child_prefix = prefix + ("    " if is_last else "│   ")
            children = node.children
            for i, child in enumerate(node.children):
                if isinstance(node.state, AgenticState) and node.state.action_request:
                    self._append_ascii_action(child, child_prefix, i == len(children) - 1, tree_lines, current)
                else:
                    self._append_ascii_node(child, child_prefix, i == len(children) - 1, tree_lines, current)

    def save_ascii_tree(self, ascii_tree: str, filename: str):
        """
        Save the ASCII representation of the tree to a file.

        Args:
            ascii_tree (str): The ASCII representation of the tree.
            filename (str): The path where the file should be saved.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Write the ASCII tree to the file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ascii_tree)

            logger.info(f"ASCII tree saved successfully to {filename}")
        except Exception as e:
            logger.error(f"Error saving ASCII tree to {filename}: {str(e)}")

    def is_duplicate(self, node: MCTSNode, new_state: State):
        for child in node.children:
            if self.equals(new_state, child.state):
                logger.info(f"{new_state.name}:{new_state.id} is a duplicate of {child.state.name}:{child.state.id}")
                return True

        return False

    def equals(self, state: State, other: State):
        if state.name != other.name:
            return False

        if state.name in ["Finished", "Rejected"]:
            return True

        if not state.previous_state or not other.previous_state:
            return False

        if state.name != "PlanToCode":
            logger.warning(f"Comparing states {state.name} and {other.name} is not supported")
            return False

        if state.previous_state.name != other.previous_state.name:
            return False

        if state.previous_state.name == "EditCode" and state.diff:
            logger.info(f"Comparing diffs: \n\n{state.diff}\n\n and \n\n{other.diff}")
            return state.diff == other.diff

        if state.previous_state.name == "IdentifyCode" and state.previous_state.outcome.get("identified_spans"):
            state_spans = state.previous_state.outcome.get("identified_spans", [])
            other_spans = other.previous_state.outcome.get("identified_spans", [])

            state_spans_set = {(span['file_path'], frozenset(span['span_ids'])) for span in state_spans}
            other_spans_set = {(span['file_path'], frozenset(span['span_ids'])) for span in other_spans}

            logger.info(f"Comparing identified spans: {state_spans_set} and {other_spans_set}")

            return state_spans_set == other_spans_set

        previous_plan_state = state.get_previous_state(state)
        other_previous_plan_state = other.get_previous_state(other)
        if previous_plan_state and previous_plan_state.action_request and other_previous_plan_state and other_previous_plan_state.action_request:
            logger.info(
                f"Comparing actions: {previous_plan_state.action_request.action} and {other_previous_plan_state.action_request.action}")
            this_dict = previous_plan_state.action_request.action.model_dump(exclude={"scratch_pad"})
            other_dict = other_previous_plan_state.action_request.action.model_dump(exclude={"scratch_pad"})
            return this_dict == other_dict

        logger.warning(f"State {state.name}:{state.id} and {state.name}:{state.id} is not equal")
        return False

    def export_nodes(self, filename: str):
        """
        Export the nodes information to a JSON file.

        Args:
            filename (str): The path where the JSON file should be saved.
        """

        def node_to_dict(node):
            return {
                'id': node.id,
                'state_name': node.state.name,
                'state_id': node.state.id,
                'visits': node.visits,
                'value': node.value,
                'raw_value': node.raw_value,
                'duplicate': node.duplicate,
                'parent_id': node.parent.id if node.parent else None,
                'children_ids': [child.id for child in node.children],
                'depth': node.get_depth(),
                'exploitation': getattr(node, 'exploitation', 0),
                'exploration': getattr(node, 'exploration', 0),
                'depth_penalty': getattr(node, 'depth_penalty', 0),
                'high_value_parent_bonus': getattr(node, 'high_value_parent_bonus', 0),
                'high_value_leaf_bonus': getattr(node, 'high_value_leaf_bonus', 0),
                'high_value_bad_children_bonus': getattr(node, 'high_value_bad_children_bonus', 0),
                'high_value_child_penalty': getattr(node, 'high_value_child_penalty', 0),
                'uct_score': self.uct_score(node.parent, node) if node.parent else 0,
                'mean_traj_reward': node.get_mean_traj_reward(),
                'resolved': getattr(node, 'resolved', False),
            }

        nodes_data = {str(node_id): node_to_dict(node) for node_id, node in self.nodes.items()}

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, indent=2)
            logger.info(f"Nodes information exported successfully to {filename}")
        except Exception as e:
            logger.error(f"Error exporting nodes information to {filename}: {str(e)}")

    def evaluate(self, node):
        workspace = self.loop.workspace
        state = node.state

        if state.output is None:
            state.output = {}

        # Always try to get the git diff
        if isinstance(workspace.file_repo, GitRepository):
            try:
                diff = workspace.file_repo.diff()
                state.output["diff"] = diff
            except Exception as e:
                logger.exception(f"Error calculating diff: {e}")

        # Run verifier if conditions are met
        if (isinstance(state, Rejected) or isinstance(state, Finished)) and workspace.verifier:
            if isinstance(workspace.file_repo, GitRepository):
                try:
                    from moatless.verify.testbed import TestbedVerifier
                    if isinstance(workspace.verifier, TestbedVerifier):
                        logger.info(f"Running evaluation on finished trajectory Node{node.id}")
                        result = workspace.verifier.evaluate()
                        if result:
                            logger.info(f"Node{node.id} Resolved: {result.resolved}")
                            state.output["evaluation_result"] = result.model_dump()
                            state.output["resolved"] = result.resolved
                            node.resolved = result.resolved

                        self.loop._trajectory.save_state(state)
                except ImportError as e:
                    logger.info(f"Error importing TestbedVerifier. Can't run evaluation. {e}")
                except Exception as e:
                    logger.exception(f"Error running evaluation. {e}")
                    state.output["error"] = traceback.format_exc()


def color_red(text: Any) -> str:
    return f"\033[91m{text}\033[0m"


def color_green(text: Any) -> str:
    return f"\033[92m{text}\033[0m"


def color_yellow(text: Any) -> str:
    return f"\033[93m{text}\033[0m"


def color_white(text: Any) -> str:
    return f"\033[97m{text}\033[0m"

