from itertools import groupby

import pytest
from regawa import BaseModel
from regawa.data import fn_groundobs_to_heterograph, render_lifted
from regawa.data.graph.graph_func import fn_action_mask_from_groundings
from regawa.data.obs.obs_func import fn_idx_obs
from regawa.model import (
    GenericModel,
    GroundObs,
    NullConst,
    check_model,
    model_to_json,
)


def create_obs(model: BaseModel, rddl_obs: GroundObs):
    graph = fn_groundobs_to_heterograph(
        model,
        stacking=False,
    )(rddl_obs)

    obs = fn_idx_obs(model)(graph)

    return obs, graph


def test_model_check(test_model: BaseModel):
    assert check_model(test_model)


def test_state_dependent_action_mask(test_model: BaseModel):
    valid_actions = {
        ("pickup", "block1"),
        ("pickup", "block2"),
        ("NOP",),
    }
    object_to_idx = {"block1": 0, "block2": 1, "block3": 2, "table1": 3, "table2": 4}

    mask_fn = fn_action_mask_from_groundings(test_model)
    action_mask = mask_fn(valid_actions, object_to_idx)

    assert action_mask.shape == (len(object_to_idx), len(test_model.action_fluents))
    assert (
        action_mask[object_to_idx["block1"], test_model.action_to_idx("pickup")] == True
    )
    assert (
        action_mask[object_to_idx["block2"], test_model.action_to_idx("pickup")] == True
    )
    assert (
        action_mask[object_to_idx["block3"], test_model.action_to_idx("pickup")]
        == False
    )
    assert (
        action_mask[object_to_idx["table1"], test_model.action_to_idx("pickup")]
        == False
    )
    assert (
        action_mask[object_to_idx["table2"], test_model.action_to_idx("pickup")]
        == False
    )
    assert action_mask[object_to_idx["block1"], test_model.action_to_idx("NOP")] == True
    assert action_mask[object_to_idx["block2"], test_model.action_to_idx("NOP")] == True
    assert action_mask[object_to_idx["block3"], test_model.action_to_idx("NOP")] == True
    assert action_mask[object_to_idx["table1"], test_model.action_to_idx("NOP")] == True
    assert action_mask[object_to_idx["table2"], test_model.action_to_idx("NOP")] == True

    pass


def test_sample_obs(test_model: BaseModel):
    rddl_obs = {
        ("at", "block1", "table1"): True,
        ("at", "block2", "table2"): True,
        ("on", "block1", "block2"): False,
        ("weight", "block1"): 1.0,
        ("weight", "block2"): 2.0,
        ("weight", "block3"): 3.0,
    }

    _, graph = create_obs(test_model, rddl_obs)

    assert graph.boolean.factors == graph.numeric.factors

    assert set(graph.boolean.factors.names) == {
        NullConst.id,
        "block1",
        "block3",
        "table2",
        "block2",
        "table1",
    }

    # since multisets are not in python, count the occurrences of each type
    counts = groupby(sorted(graph.boolean.factors.types))
    type_counts = {k: len(list(v)) for k, v in counts}
    assert type_counts == {
        NullConst.type: 1,
        "block": 3,
        "table": 2,
    }

    assert set(graph.boolean.variables.values) == {True, False}
    assert set(graph.numeric.variables.values) == {1.0, 3.0, 2.0}
    assert set(graph.boolean.variables.types) == {"at", "on"}
    assert set(graph.numeric.variables.types) == {"weight"}

    pass


def test_render_lifted(test_model: BaseModel):
    graph = render_lifted(test_model)

    with open("test_lifted.dot", "w") as f:
        f.write(graph)
    assert graph is not None


def test_serialization(test_model: BaseModel):
    model = test_model
    json_data = model_to_json(model)

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
