from regawa.rddl import register_env
import gymnasium as gym
import pytest


@pytest.fixture(scope="module")
def tiger_env():
    domain = "rddl/tiger/domain.rddl"
    instance = "rddl/tiger/instance_1.rddl"
    env_id = register_env()
    env = gym.make(
        env_id,
        domain=domain,
        instance=instance,
        remove_false=True,
    )
    return env


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (("open", "left"), 10.0),  # open left door
        (("open", "right"), -100.0),  # open right door
    ],
)
def test_open(tiger_env, action, expected_reward):
    obs, info = tiger_env.reset()

    print("Info:", info["rddl_state"])

    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

    print("Info:", info["rddl_state"])

    assert (
        terminated is True
    ), "The episode should be terminated after opening the door."
    assert (
        reward == expected_reward
    ), f"Expected reward {expected_reward}, but got {reward}."


@pytest.mark.parametrize(
    "action, expect_hear",
    [
        (
            ("listen", "left"),
            False,
        ),  # we do not expect to hear the tiger from the left door
        (
            ("listen", "right"),
            True,
        ),  # we expect to hear the tiger from the right door
    ],
)
def test_listen(tiger_env, action, expect_hear):
    obs, info = tiger_env.reset(seed=22)

    print("Info:", info["rddl_state"])

    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

    print("Info:", info["rddl_state"])

    assert terminated is False, "The episode should not be terminated after listening."

    if expect_hear:
        assert ("growl", "right") in info[
            "rddl_state"
        ], "Did not expect to hear the tiger."
        assert ("growl", "left") not in info[
            "rddl_state"
        ], "Did not expect to hear the tiger on the left side."
    else:
        assert ("growl", "right") not in info[
            "rddl_state"
        ], "Expected to hear the tiger."
        assert ("growl", "left") not in info[
            "rddl_state"
        ], "Expected to hear the tiger."


def test_false_positive(tiger_env):
    obs, info = tiger_env.reset(seed=22)

    action = (
        "listen",
        "left",
    )  # though the tiger is on the right, we can observe a false positive from the left door if we keep listening
    predicate = info["action_fluents"].index(action[0])
    door = info["idx_to_object"].index(action[1])

    heard_tiger = False
    done = False
    while not done:
        obs, reward, terminated, truncated, info = tiger_env.step((predicate, door))

        assert reward == -1.0
        print("Info:", info["rddl_state"])

        if ("growl", "left") in info["rddl_state"]:
            heard_tiger = True
            break

        done = terminated or truncated

    assert heard_tiger, "Did not hear the tiger from the left door"


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
