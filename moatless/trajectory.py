import json
import logging
from typing import Any, Optional, List
from datetime import datetime, UTC

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from moatless.workspace import Workspace
from moatless.transition_rules import TransitionRules
from moatless.state import AgenticState, get_state_class, State, Content, StateOutcome, ActionTransaction
from moatless.schema import (
    Completion, Usage,
)

logger = logging.getLogger(__name__)


class TrajectoryState(BaseModel):
    id: int
    snapshot: Optional[dict] = None
    state: State
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def name(self):
        return self.state.name if self.state else None

    def model_dump(self, **kwargs):
        data = {
            "id": self.id,
            "name": self.state.name,
            "created_at": self.created_at.isoformat(),
        }

        if self.snapshot:
            data["snapshot"] = self.snapshot

        if self.state.previous_state:
            data["previous_state_id"] = self.state.previous_state.id

        if self.state.origin_state:
            data["origin_state_id"] = self.state.origin_state.id

        # Handle the exclude parameter for the state properties
        state_kwargs = kwargs.copy()
        if "exclude" in state_kwargs:
            if isinstance(state_kwargs["exclude"], set):
                state_kwargs["exclude"] = state_kwargs["exclude"].union({"id"})
            elif isinstance(state_kwargs["exclude"], dict):
                state_kwargs["exclude"] = {**state_kwargs["exclude"], "id": True}
        else:
            state_kwargs["exclude"] = {"id"}

        properties = self.state.model_dump(**state_kwargs) if self.state else None
        if properties:
            data["properties"] = properties

        if isinstance(self.state, AgenticState) and self.state.actions:
            data["actions"] = [a.model_dump(**kwargs) for a in self.state.actions]

        return data


class Trajectory:
    def __init__(
        self,
        name: str,
        workspace: Workspace,
        initial_message: Optional[str] = None,
        persist_path: Optional[str] = None,
        transition_rules: Optional[TransitionRules] = None,
    ):
        self._name = name
        self._persist_path = persist_path
        self._initial_message = initial_message
        self._workspace = workspace

        # Workaround to set to keep the current initial workspace state when loading an existing trajectory.
        # TODO: Remove this when we have a better way to handle this.
        self._initial_workspace_state = self._workspace.dict()

        self._transition_rules = transition_rules

        self._current_transition_id = 0
        self._transitions: dict[int, TrajectoryState] = {}

        self._info: dict[str, Any] = {}
        self._created_at = datetime.now(UTC)

    @classmethod
    def load(cls, file_path: str, skip_workspace: bool = False, **kwargs):
        logger.info(f"Loading trajectory from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)

        if "transition_rules" in data:
            try:
                transition_rules = TransitionRules.model_validate(data["transition_rules"])
            except Exception as e:
                logger.exception(f"Error loading transition rules. Will still load trajectory.")
                transition_rules = None
        else:
            transition_rules = None

        if skip_workspace:
            workspace = Workspace(file_repo=None, file_context=None)
        else:
            if data.get("workspace", {}).get("code_index") and not data["workspace"]["code_index"].get("index_name") and data["info"].get("instance_id"):
                data["workspace"]["code_index"] = {"index_name": data["info"]["instance_id"]}

            workspace = Workspace.from_dict(data["workspace"], **kwargs)

        trajectory = cls(
            name=data["name"],
            initial_message=data["initial_message"],
            persist_path=file_path,
            transition_rules=transition_rules,
            workspace=workspace,
        )
        trajectory._created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))

        trajectory._info = data.get("info", {})

        trajectory._transitions = {}
        trajectory._current_transition_id = data.get("current_transition_id", 0)

        for t in data["transitions"]:
            try:
                trajectory_state = Trajectory._map_state(t, trajectory)
                trajectory._transitions[t["id"]] = trajectory_state

                if t["id"] == trajectory._current_transition_id and not skip_workspace:
                    workspace.file_context.restore_from_snapshot(t["snapshot"]["file_context"])

            except Exception as e:
                logger.exception(f"Error loading state {t.get('name')} {t.get('id')}: {e}")
                raise e

        # Set previous_state and next_states
        for t in data["transitions"]:
            try:
                current_state = trajectory._transitions[t["id"]].state
                if t.get("previous_state_id") is not None:
                    current_state.previous_state = trajectory._transitions.get(
                        t["previous_state_id"]
                    ).state
                    current_state.previous_state.next_states.append(current_state)
                
            except KeyError as e:
                logger.exception(
                    f"Missing key {e}, existing keys: {trajectory._transitions.keys()}"
                )
                raise

            try:
                current_state = trajectory._transitions[t["id"]].state
                if t.get("origin_state_id") is not None and t["origin_state_id"] != t["id"]:
                    current_state.origin_state = trajectory._transitions.get(
                        t["origin_state_id"]
                    ).state
                    current_state.origin_state.clones.append(current_state)
            except KeyError as e:
                logger.exception(
                    f"Missing key {e}, existing keys: {trajectory._transitions.keys()}"
                )
                raise

        trajectory._info = data.get("info", {})
        logger.info(
            f"Loaded trajectory {trajectory._name} with {len(trajectory._transitions)} transitions"
        )

        return trajectory

    @staticmethod
    def _map_state(t: dict, trajectory: "Trajectory"):
        state_class = get_state_class(t["name"])
        state_data = t.get("properties", {})
        state_data["id"] = t["id"]
        state = state_class.model_validate(state_data)

        state._workspace = trajectory._workspace
        state._initial_message = trajectory._initial_message
        state._actions = []
        if "actions" in t:
            for a in t["actions"]:
                try:
                    if state.action_type() is None:
                        request = Content.model_validate(a["request"])
                    else:
                        request = state.action_type().model_validate(a["request"])
                    response = StateOutcome.model_validate(a.get("response"))
                    if a.get("completion"):
                        completion = Completion.model_validate(a.get("completion"))
                    else:
                        completion = None
                    state._actions.append(
                        ActionTransaction(
                            request=request,
                            response=response,
                            completion=completion,
                        )
                    )
                except Exception as e:
                    logger.exception(
                        f"Error loading action in state {state.name}:{state.id}: {a}"
                    )
                    raise e

        return TrajectoryState(
            id=t["id"],
            snapshot=t.get("snapshot"),
            state=state,
            created_at=datetime.fromisoformat(t.get("created_at", datetime.utcnow().isoformat())),
        )

    @property
    def initial_message(self):
        return self._initial_message

    @property
    def info(self):
        return self._info

    @property
    def states(self) -> List[dict]:
        return [t.state.model_dump() for t in self.transitions]

    @property
    def transition_rules(self) -> TransitionRules:
        return self._transition_rules

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def transitions(self) -> List[TrajectoryState]:
        return sorted(self._transitions.values(), key=lambda x: x.id)

    def set_current_state(self, state: State | int | None = None):
        if isinstance(state, int):
            self._current_transition_id = state
        elif state is not None:
            self._current_transition_id = state.id
        else:
            raise ValueError("Either state or state_id must be provided")
        self._maybe_persist()

    def get_current_state(self) -> State:
        return self._transitions.get(self._current_transition_id).state

    def update_workspace_to_current_state(self):
        self.restore_from_snapshot(self._transitions[self._current_transition_id])

    def restore_from_snapshot(self, state: TrajectoryState):
        if not state.snapshot:
            logger.info(
                f"restore_from_snapshot(state: {state.id}:{state.name}) No snapshot found"
            )
            return

        logger.info(
            f"restore_from_snapshot(state: {state.id}:{state.name}) Restoring from snapshot"
        )

        if state.snapshot.get("repository"):
            self._workspace.file_repo.restore_from_snapshot(
                state.snapshot["repository"]
            )

        if state.snapshot.get("file_context"):
            self._workspace.file_context.restore_from_snapshot(
                state.snapshot["file_context"]
            )

    def get_snapshot(self, state_id: int) -> dict:
        return self._transitions[state_id].snapshot

    def save_state(self, state: State):
        if state.id in self._transitions:
            self._transitions[state.id].state = state
        else:
            snapshot = state.workspace.snapshot() if state.workspace else None
            transition = TrajectoryState(
                id=state.id,
                state=state,
                snapshot=snapshot,
                created_at=datetime.now(UTC)
            )
            self._transitions[state.id] = transition

        self._maybe_persist()

    def get_state(self, state_id: int) -> State | None:
        if state_id in self._transitions:
            return self._transitions[state_id].state
        return None

    def save_info(self, info: dict):
        self._info.update(info)
        self._maybe_persist()

    def get_mocked_actions(self) -> List[dict]:
        """
        Return a list of actions that can be used to mock the trajectory.
        """
        actions = []

        for transition in self.transitions:
            if isinstance(transition.state, AgenticState):
                for action in transition.state.actions:
                    actions.append(action.request.model_dump())
        return actions

    def get_expected_states(self) -> List[str]:
        """
        Return a list of expected states in the trajectory to use for verification when rerunning the trajectory.
        """
        return [transition.state.name for transition in self.transitions[1:]]

    def get_states_by_name(self, state_name: str) -> List[State]:
        return [t.state for t in self.transitions if t.state.name == state_name]

    def to_dict(self, **kwargs):
        return {
            "name": self._name,
            "transition_rules": self._transition_rules.model_dump(exclude_none=True)
            if self._transition_rules
            else None,
            "workspace": self._initial_workspace_state,
            "initial_message": self._initial_message,
            "current_transition_id": self._current_transition_id,
            "transitions": [t.model_dump(exclude_none=True, **kwargs) for t in self.transitions],
            "info": self._info,
            "created_at": self._created_at.isoformat(),
        }

    def total_usage(self) -> Usage:
        total_usage = Usage()
        for transition in self._transitions.values():
            if not hasattr(transition.state, "total_usage"):
                continue

            total_usage += transition.state.total_usage()

        return total_usage

    def _maybe_persist(self):
        if self._persist_path:
            self.persist(self._persist_path)

    def persist(self, file_path: str, **kwargs):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self.to_dict(**kwargs),
                    indent=2,
                    default=to_jsonable_python,
                )
            )


if "__main__" == __name__:
    traj = Trajectory.load("/home/albert/repos/albert/sw-planner-2/trajs/evaluations/20240916_django_gpt-4o-mini-2024-07-18_max_exp2_mcts_True_debate_False_provide_feedback_False_temp_bias_0.0_eval_name_None/django__django-11039/trajectory.json", skip_workspace=True)

    print(traj._transitions.get(22).snapshot["file_context"])