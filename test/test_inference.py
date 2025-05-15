from regawa import GNNParams, ActionMode, agent_from_model
from regawa.rddl import model_from_domain
import numpy as np
from regawa.inference import fn_get_agent_output
import torch as th

from regawa.wrappers import remove_false


def test_inference():
    lifted_model, _ = model_from_domain(
        "rddl/conditional_bandit/domain.rddl",
        "rddl/conditional_bandit/instance_1.rddl",
    )
    action_mode = ActionMode.ACTION_THEN_NODE

    params = GNNParams(
        layers=4,
        embedding_dim=16,
        activation=th.nn.Mish(),
        aggregation="max",
        action_mode=action_mode,
    )

    agent = agent_from_model(lifted_model, params)

    test_obs = {
        ("light", "r_m"): np.True_,
        ("light", "g_m"): np.False_,
        ("CONNECTED", "red", "r_m"): np.True_,
        ("CONNECTED", "red", "g_m"): np.False_,
        ("CONNECTED", "green", "r_m"): np.False_,
        ("CONNECTED", "green", "g_m"): np.True_,
        ("PAYOUT", "r_m"): np.float64(1.0),
        ("PAYOUT", "g_m"): np.float64(1.0),
        ("LIGHT_PROB",): np.float64(0.5),
    }

    # add_constants = add_constants_fn(
    #     grounded_model,
    # )

    agent_output = fn_get_agent_output(agent, lifted_model, remove_false, action_mode)

    action, weight_by_object, weight_by_action, graph = agent_output(test_obs)

    pass


if __name__ == "__main__":
    test_inference()
