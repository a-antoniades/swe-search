import logging

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Any, Type, Optional, List

from moatless.settings import Settings
from moatless.state import AgenticState, get_state_class, State, Rejected
from moatless.workspace import Workspace


logger = logging.getLogger(__name__)


class TransitionRule(BaseModel):
    trigger: str = Field(
        ...,
        description="The trigger from the current state that causes the transition to fire.",
    )
    source: type[State] = Field(
        ..., description="The source state that the transition rule is defined for."
    )
    dest: type[State] = Field(
        ...,
        description="The destination state that the transition rule is defined for.",
    )
    required_fields: Optional[set[str]] = Field(
        default=None,
        description="The fields that are required for the transition to fire.",
    )
    excluded_fields: Optional[set[str]] = Field(
        default=None, description="The fields that are excluded from the transition."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to set on destination state.",
    )

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["source"] = self.source.__name__
        data["dest"] = self.dest.__name__

        if data.get("required_fields"):
            data["required_fields"] = list(data.get("required_fields"))

        if data.get("excluded_fields"):
            data["excluded_fields"] = list(data.get("excluded_fields"))

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_state_classes(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("source"), str):
                data["source"] = get_state_class(data["source"])
            if isinstance(data.get("dest"), str):
                data["dest"] = get_state_class(data["dest"])

        return data


class AgenticLoopSettings(BaseModel):
    max_cost: float = Field(
        0.5,
        description="The maximum total cost of LLM completion requests.",
    )

    max_transitions: int = Field(
        25,
        description="The maximum number of transitions to run the loop.",
    )



class TreeSearchSettings(BaseModel):
    max_expansions: int = Field(
        3,
        description="The maximum number of expansions of one state.",
    )

    max_iterations: int = Field(
        100,
        description="The maximum number of iterations to run the tree search.",
    )

    min_finished_transitions_for_early_stop: Optional[int] = Field(
        None,
        description="The minimum number of finished transitions to consider before finishing.",
    )

    max_finished_transitions: Optional[int] = Field(
        None,
        description="The maximum number of finished transitions to consider before finishing",
    )

    reward_threshold: Optional[int] = Field(
        101,
        description="The reward threshold to consider before finishing.",
    )

    states_to_explore: List[str] = Field(
        ["SearchCode", "PlanToCode"],
        description="The states to explore.",
    )

    provide_feedback: bool = Field(
        False,
        description="Whether to provide feedback from previosly evaluated transitions.",
    )

    debate: bool = Field(
        False,
        description="Whether to debate the rewards to transitions.",
    )

    value_function_model: Optional[str] = Field(
        None,
        description="The model to use for the value function.",
    )

    best_first: bool = Field(
        True,
        description="Whether to use best first search.",
    )
    value_function_model_temperature: Optional[float] = Field(
        0.2,
        description="The temperature to use for the value function model.",
    )


class TransitionRules(BaseModel):

    loop_settings: AgenticLoopSettings | None = Field(
        None,
        description="The settings for the loop.",
    )

    tree_search_settings: TreeSearchSettings | None = Field(
        None,
        description="The settings for the tree search.",
    )

    initial_state: type[State] | None = Field(
        default=None,
        description="The initial state for the loop.",
        deprecated="Initial state should be set in transition_rules instead.",
    )
    transition_rules: list[TransitionRule] = Field(
        ..., description="The transition rules for the loop."
    )
    global_params: dict[str, Any] = Field(
        default_factory=dict, description="Global parameters used by all transitions."
    )
    state_params: dict[type[State], dict[str, Any]] = Field(
        default_factory=dict, description="State-specific parameters."
    )

    _source_trigger_index: dict[tuple[type[State], str], list[TransitionRule]] = (
        PrivateAttr(default_factory=dict)
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._build_source_trigger_index()

    def model_dump(self, **kwargs):
        data = {
            "global_params": self.global_params,
            "state_params": {k.__name__: v for k, v in self.state_params.items()},
            "loop_settings": self.loop_settings.model_dump() if self.loop_settings else None,   
            "tree_search_settings": self.tree_search_settings.model_dump() if self.tree_search_settings else None,
            "transition_rules": [
                rule.model_dump(**kwargs) for rule in self.transition_rules
            ],
        }

        if self.initial_state:
            data["initial_state"] = self.initial_state.__name__

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_before_init(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if isinstance(data.get("initial_state"), str):
                data["initial_state"] = get_state_class(data["initial_state"])

            if "state_params" in data:
                data["state_params"] = {
                    get_state_class(k) if isinstance(k, str) else k: v
                    for k, v in data["state_params"].items()
                }

        if "global_params" not in data:
            data["global_params"] = {}

        if "model" not in data["global_params"]:
            logger.info(
                f"No model specified in global_params. Using default model: {Settings.default_model}"
            )
            data["global_params"]["model"] = Settings.default_model

        return data

    def _build_source_trigger_index(self):
        for rule in self.transition_rules:
            key = (rule.source, rule.trigger)
            if key not in self._source_trigger_index:
                self._source_trigger_index[key] = []
            self._source_trigger_index[key].append(rule)

    def find_transition_rule_by_source_and_trigger(
        self, source: type[State], trigger: str
    ) -> list[TransitionRule]:
        return self._source_trigger_index.get((source, trigger), [])

    def params(self, rule: TransitionRule) -> dict[str, Any]:
        params = {}
        params.update(self.global_params)
        params.update(self.state_params.get(rule.dest, {}))
        params.update(rule.params)
        return params

    def get_next_rule(
        self, source: State, trigger: str | None = None, data: dict[str, Any] = {}
    ) -> TransitionRule | None:
        if trigger == "init" and self.initial_state:
            logger.warning(
                "Using deprecated 'initial_state'. Set initial state in transition_rules instead."
            )
            return TransitionRule(
                trigger="init",
                source=source.__class__,
                dest=self.initial_state,
            )

        if trigger is None:
            return TransitionRule(
                trigger="continue",
                source=source.__class__,
                dest=source.__class__,
            )

        transition_rules = self.find_transition_rule_by_source_and_trigger(
            source.__class__, trigger
        )

        if not transition_rules and trigger == "reject":
            logger.warning(f"No transition rule found for {source} with trigger {trigger}, will default to Rejected transition")
            return TransitionRule(
                trigger="reject",
                source=source.__class__,
                dest=Rejected
            )

        for transition_rule in transition_rules:
            if (
                transition_rule.required_fields
                and not transition_rule.required_fields.issubset(data.keys())
            ):
                logger.info(f"Missing required fields for transition {transition_rule}")
                continue

            return transition_rule

        return None
