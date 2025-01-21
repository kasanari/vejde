from regawa.rddl import RDDLModel
from regawa import BaseModel
import regawa.model.utils as utils
import pytest


@pytest.fixture()
def model():
    import pyRDDLGym

    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)  # type: ignore
    return RDDLModel(env.model)


def test_num_fluents(model: BaseModel):
    assert model.num_fluents == 6


def test_num_types(model: BaseModel):
    assert model.num_types == 3


def test_num_actions(model: BaseModel):
    assert model.num_actions == 2


def test_variable_range(model: BaseModel):
    f_r = model.fluent_range

    for r in model.fluents:
        assert f_r(r)

    assert f_r("light") is bool
    assert f_r("PAYOUT") is float
    assert f_r("CONNECTED") is bool
    assert f_r("None") is bool


def test_fluent_params(model: BaseModel):
    f_p = model.fluent_params

    for r in model.fluents:
        assert f_p(r) is not None

    assert f_p("light") == ("machine",)
    assert f_p("PAYOUT") == ("machine",)
    assert f_p("CONNECTED") == ("button", "machine")
    assert f_p("None") == ()


def test_fluent_param(model: BaseModel):
    f_p = model.fluent_param
    f_s = model.fluent_params

    params = f_s("CONNECTED")

    assert params[0] == f_p("CONNECTED", 0)
    assert params[1] == f_p("CONNECTED", 1)


def test_fluents_of_arity(model: BaseModel):
    fluents_of_arity = utils.fluents_of_arity(model)

    assert fluents_of_arity(1) == (
        "PAYOUT",
        "light",
        "press",
    )

    assert fluents_of_arity(2) == ("CONNECTED",)


def test_groundings(model: BaseModel):
    """
    A list of all possible grounded variables in the language.
    relation___object1__object2__...__objectN
    """
    ...
    assert len(model.groundings) == 9


def test_action_fluents(model: BaseModel):
    """relations/fluents/predicates that can be used as actions in the model."""
    assert len(model.action_fluents) == 2


def test_action_groundings(model: BaseModel):
    """groundings of action fluents/variables.
    one the form: relation___object1__object2__...__objectN
    """
    assert len(model.action_groundings) == 3


def test_num_relations(model: BaseModel):
    """The number of relations/predicates in the model. This includes nullary and unary relations, which may also be called constants and attributes."""
    ...
    assert model.num_fluents == len(model.fluents)


def test_type_to_idx(model: BaseModel):
    """
    A mapping from object type to an index.
    This should be consistent across all instances of the same domain.
    Note that 0 is reserved for padding.
    """
    for t in model.types:
        assert model.type_to_idx(t) is not None

    assert model.type_to_idx("None") == 0


def test_idx_to_type(model: BaseModel):
    """
    A mapping from an index to an object type.
    This should be consistent across all instances of the same domain.
    Note that 0 is reserved for padding.
    """
    for i in range(model.num_types):
        assert model.idx_to_type(i) is not None

    assert model.idx_to_type(0) == "None"


def test_rel_to_idx(model: BaseModel):
    """
    A mapping from a relation/predicate to an index.
    This should be consistent across all instances of the same domain.
    Note that 0 is reserved for padding.
    """
    for r in model.fluents:
        assert model.fluent_to_idx(r) is not None

    assert model.fluent_to_idx("None") == 0


def test_idx_to_relation(model: BaseModel):
    """
    A mapping from an index to a relation/predicate.
    This should be consistent across all instances of the same domain.
    Note that 0 is reserved for padding.
    """
    for i in range(model.num_fluents):
        assert model.idx_to_fluent(i) is not None

    assert model.idx_to_fluent(0) == "None"


def test_idx_to_action(model: BaseModel):
    for i in range(model.num_actions):
        assert model.idx_to_action(i) is not None

    assert model.idx_to_action(0) == "None"


def test_action_to_idx(model: BaseModel):
    for a in model.action_fluents:
        assert model.action_to_idx(a) is not None

    assert model.action_to_idx("None") == 0


def test_arity(model: BaseModel):
    assert model.arity("press") == 1
    assert model.arity("light") == 1
    assert model.arity("PAYOUT") == 1
    assert model.arity("CONNECTED") == 2
