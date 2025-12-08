from regawa import Grounding, GroundingRange
import json

def obs_to_json_friendly_obs(
    obs: dict[Grounding, GroundingRange],
) -> dict[str, GroundingRange]:
    return {"__".join(k): v for k, v in obs.items()}


def json_friendly_obs_to_obs(
    obs: dict[str, GroundingRange],
) -> dict[Grounding, GroundingRange]:
    return {tuple(k.split("__")): v for k, v in obs.items()}

def obs_from_json(obs_json: str) -> dict[Grounding, GroundingRange]:
    obs_dict = json.loads(obs_json)
    return json_friendly_obs_to_obs(obs_dict)


def obs_to_json(obs: dict[Grounding, GroundingRange]) -> str:
    json_friendly_dict = obs_to_json_friendly_obs(obs)
    return json.dumps(json_friendly_dict)


def step_to_json(
    obs: dict[Grounding, GroundingRange], action: dict[Grounding, GroundingRange]
) -> str:
    json_friendly_step = {
        "obs": obs_to_json_friendly_obs(obs),
        "action": obs_to_json_friendly_obs(action),
    }
    return json.dumps(json_friendly_step)


def step_from_json(step_json: str) -> dict[str, dict[Grounding, GroundingRange]]:
    step_dict = json.loads(step_json)
    return {
        "obs": json_friendly_obs_to_obs(step_dict["obs"]),
        "action": json_friendly_obs_to_obs(step_dict["action"]),
    }