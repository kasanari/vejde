from regawa import BaseModel, BaseGroundedModel


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

    assert model.fluents[0] == "None", "First fluent must be 'None' for padding."
    assert (
        model.action_fluents[0] == "None"
    ), "First action fluent must be 'None' for padding."
    assert model.types[0] == "None", "First type must be 'None' for padding."
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
