from . import BaseGroundedModel, BaseModel
from .null import NullConst


def _check_model(model: BaseModel) -> bool:
    assert isinstance(
        model, BaseModel
    ), "Provided model is not an instance of BaseModel"

    assert model.num_types > 0, "Model must have at least one type."
    assert model.num_actions > 0, "Model must have at least one action fluent."
    assert model.num_fluents > 0, "Model must have at least one fluent."

    assert (
        len(model.fluents) == model.num_fluents
    ), "Fluents length does not match num_fluents."

    assert (
        model.fluents[NullConst.idx] == NullConst.action
    ), "First fluent must be 'None' for padding."
    assert (
        model.action_fluents[NullConst.idx] == NullConst.action
    ), "First action fluent must be 'None' for padding."
    assert (
        model.types[NullConst.idx] == NullConst.type
    ), "First type must be 'None' for padding."

    for i, fluent in enumerate(model.fluents):
        assert (
            model.fluent_to_idx(fluent) == i
        ), f"Fluent '{fluent}' does not map to its correct index {i}."
        assert (
            model.idx_to_fluent(i) == fluent
        ), f"Index {i} does not map back to fluent '{fluent}'."

    for i, action in enumerate(model.action_fluents):
        assert (
            model.action_to_idx(action) == i
        ), f"Action fluent '{action}' does not map to its correct index {i}."
        assert (
            model.idx_to_action(i) == action
        ), f"Index {i} does not map back to action fluent '{action}'."

    for i, obj_type in enumerate(model.types):
        assert (
            model.type_to_idx(obj_type) == i
        ), f"Type '{obj_type}' does not map to its correct index {i}."
        assert (
            model.idx_to_type(i) == obj_type
        ), f"Index {i} does not map back to type '{obj_type}'."

    return True


def check_model(model: BaseModel) -> bool:
    try:
        return _check_model(model)
    except Exception as e:
        print(f"Model check failed: {e}")
        return False


def _check_grounded_model(model: BaseModel, grounded_model: BaseGroundedModel) -> bool:
    assert isinstance(
        grounded_model, BaseGroundedModel
    ), "Provided grounded model is not an instance of BaseGroundedModel"

    assert (
        grounded_model.groundings == model
    ), "Grounded model does not match the base model."

    assert grounded_model.action_groundings == model.action_fluents
    return True


def check_grounded_model(model: BaseModel, grounded_model: BaseGroundedModel) -> bool:
    try:
        return _check_grounded_model(model, grounded_model)
    except Exception as e:
        print(f"Grounded model check failed: {e}")
        return False
