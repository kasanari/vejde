from typing import Any
from regawa import BaseModel
from functools import cached_property, cache
import pytest
from regawa.model import BaseGroundedModel, GroundObs, Grounding
from regawa.model import check_model
from regawa.model.generic_model import GenericModel
from regawa.model.utils import to_json
from regawa.wrappers.graph_utils import fn_groundobs_to_heterograph
from regawa.wrappers.index_obs_wrapper import fn_idx_obs
from regawa.model.null import NullConst
from regawa.wrappers.render_utils import render_lifted


class TestModel(BaseModel):
    """Sample model for testing purposes. Loosely based on block stacking problems."""

    _types = (NullConst.type, "block", "table")
    _fluents = (NullConst.action, "at", "on", "weight", "pickup", "put")
    _actions = (NullConst.action, "pickup", "put")

    _params = {
        NullConst.action: (),
        "at": ("block", "table"),
        "on": ("block", "block"),
        "pickup": ("block",),
        "put": ("block", "table"),
        "weight": ("block",),
    }

    _ranges = {
        NullConst.action: bool,
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
        try:
            return self._params[fluent]
        except KeyError:
            raise KeyError(f"Fluent {fluent} not found in model parameters.")

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
        graph = fn_groundobs_to_heterograph(
            self._model,
            stacking=False,
        )(rddl_obs)

        obs = fn_idx_obs(self._model)(graph)

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

    assert set(graph.boolean.factors.names) == {
        NullConst.id,
        "block1",
        "block3",
        "table2",
        "block2",
        "table1",
    }

    assert set(graph.boolean.factors.types) == {
        NullConst.type,
        "block",
        "block",
        "table",
        "block",
        "table",
    }

    assert set(graph.boolean.variables.values) == {True, True, False}
    assert set(graph.numeric.variables.values) == {1.0, 3.0, 2.0}
    assert set(graph.boolean.variables.types) == {"at", "at", "on"}
    assert set(graph.numeric.variables.types) == {"weight", "weight", "weight"}

    pass


def test_render_lifted():
    graph = render_lifted(TestModel())

    with open("test_lifted.dot", "w") as f:
        f.write(graph)
    assert graph is not None


def test_serialization():
    model = TestModel()
    json_data = to_json(model)

    new_model = GenericModel.from_json(json_data)

    assert check_model(new_model)

    assert new_model.fluents == model.fluents
    assert new_model.types == model.types
    assert new_model.action_fluents == model.action_fluents
    for fluent in model.fluents:
        assert new_model.arity(fluent) == model.arity(fluent)
        assert new_model.fluent_params(fluent) == model.fluent_params(fluent)
        assert new_model.fluent_range(fluent) == model.fluent_range(fluent)
    for t in model.types:
        assert new_model.type_to_idx(t) == model.type_to_idx(t)
        assert new_model.idx_to_type(model.type_to_idx(t)) == t
    for action in model.action_fluents:
        assert new_model.action_to_idx(action) == model.action_to_idx(action)
        assert new_model.idx_to_action(model.action_to_idx(action)) == action


if __name__ == "__main__":
    # run the tests, with flags to allow debugging
    pytest.main(["-s", "-v", __file__])
