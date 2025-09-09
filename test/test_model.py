from typing import Any
from regawa import BaseModel
from functools import cached_property, cache
import pytest
from regawa.model.base_grounded_model import BaseGroundedModel, GroundObs, Grounding
from regawa.model.model_checker import check_model
from regawa.inference import fn_graph_to_obsdata, fn_groundobs_to_graph
from regawa.wrappers.render_utils import render_lifted

class TestModel(BaseModel):
    """Sample model for testing purposes. Loosely based on block stacking problems."""

    _types = ("None", "block", "table")
    _fluents = ("None", "at", "on", "weight")
    _actions = ("None", "pickup", "put")

    _params = {
        "None": (),
        "at": ("block", "table"),
        "on": ("block", "block"),
        "pickup": ("block",),
        "put": ("block", "table"),
        "weight": ("block",),
    }

    _ranges = {
        "None": bool,
        "at": bool,
        "on": bool,
        "pickup": bool,
        "put": bool,
        "weight": float,
    }

    @cached_property
    def num_types(self) -> int:
        return len(self._types)

    @cached_property
    def num_actions(self) -> int:
        return len(self._actions)

    @cache
    def fluent_range(self, fluent: str) -> type:
        return self._ranges[fluent]

    @cache
    def fluent_params(self, fluent: str) -> tuple[str, ...]:
        return self._params[fluent]

    @cache
    def fluent_param(self, fluent: str, position: int) -> str:
        return self._params[fluent][position]

    @cached_property
    def action_fluents(self) -> tuple[str, ...]:
        return self._actions

    @cached_property
    def num_fluents(self) -> int:
        return len(self._fluents)

    @cache
    def type_to_idx(self, type: str) -> int:
        return self._types.index(type)

    @cache
    def idx_to_type(self, idx: int) -> str:
        return self._types[idx]

    @cache
    def fluent_to_idx(self, relation: str) -> int:
        return self._fluents.index(relation)

    @cached_property
    def fluents(self) -> tuple[str, ...]:
        return self._fluents

    @cached_property
    def types(self) -> tuple[str, ...]:
        return self._types

    @cache
    def idx_to_fluent(self, idx: int) -> str:
        return self._fluents[idx]

    @cache
    def idx_to_action(self, idx: int) -> str:
        return self._actions[idx]

    @cache
    def action_to_idx(self, action: str) -> int:
        return self._actions.index(action)

    @cache
    def arity(self, fluent: str) -> int:
        return len(self._params[fluent])


class TestGroundedModel(BaseGroundedModel):
    _model: BaseModel = TestModel()

    _objects = (
        "block1",
        "block2",
        "block3",
        "table1",
        "table2",
    )

    _object_types: dict[str, str] = {
        "block1": "block",
        "block2": "block",
        "block3": "block",
        "table1": "table",
        "table2": "table",
    }

    _constants: GroundObs = {
        ("weight", "block1"): 1.0,
        ("weight", "block2"): 2.0,
        ("weight", "block3"): 3.0,
    }

    @cached_property
    def groundings(self) -> tuple[Grounding, ...]:
        return tuple(
            [
                (relation, *objects)
                for relation in self._model.fluents
                for objects in zip(
                    *[
                        (obj,)
                        for obj in self._objects
                        for i in range(self._model.arity(relation))
                        if self._model.fluent_param(relation, i)
                        == self._object_types[obj]
                    ]
                )
            ]
        )

    @cached_property
    def action_groundings(self) -> tuple[Grounding, ...]:
        """groundings of action fluents/variables.
        on the form: (relation, object1, object2,..., objectN)
        """
        ...

    @cached_property
    def constant_groundings(self) -> tuple[Grounding, ...]:
        """Groundings assumed to be constant in the model."""
        return (
            ("weight", "block1"),
            ("weight", "block2"),
            ("weight", "block3"),
        )

    @cache
    def constant_value(self, constant_grounding: Grounding) -> Any:
        return self._constants[constant_grounding]

    def create_obs(self, rddl_obs: GroundObs):
        graph = fn_groundobs_to_graph(self._model, lambda x: x)(rddl_obs)

        obs = fn_graph_to_obsdata(self._model)(graph)  # to ensure types are correct

        return obs, graph


def test_model_check():
    model = TestModel()

    assert check_model(model)


def test_sample_obs():
    model = TestGroundedModel()
    rddl_obs = {
        ("at", "block1", "table1"): True,
        ("at", "block2", "table2"): True,
        ("on", "block1", "block2"): False,
        ("weight", "block1"): 1.0,
        ("weight", "block2"): 2.0,
        ("weight", "block3"): 3.0,
    }

    _, graph = model.create_obs(rddl_obs)

    assert graph.boolean.factors == graph.numeric.factors

    assert set(graph.boolean.factors) == {
        "None",
        "block1",
        "block3",
        "table2",
        "block2",
        "table1",
    }

    assert set(graph.boolean.factor_types) == {
        "None",
        "block",
        "block",
        "table",
        "block",
        "table",
    }

    assert set(graph.boolean.variable_values) == {True, True, False}
    assert set(graph.numeric.variable_values) == {1.0, 3.0, 2.0}
    assert set(graph.boolean.variables) == {"at", "at", "on"}
    assert set(graph.numeric.variables) == {"weight", "weight", "weight"}

    pass

def test_render_lifted():
    graph = render_lifted(TestModel())

    with open("test_lifted.dot", "w") as f:
        f.write(graph)
    assert graph is not None

if __name__ == "__main__":
    pytest.main([__file__])
