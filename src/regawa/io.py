
from regawa import Grounding, GroundingRange


def obs_to_json_friendly_obs(
    obs: dict[Grounding, GroundingRange],
) -> dict[str, GroundingRange]:
	return {"__".join(k): v for k, v in obs.items()}

def json_friendly_obs_to_obs(
    obs: dict[str, GroundingRange],
) -> dict[Grounding, GroundingRange]:
	return {tuple(k.split("__")): v for k, v in obs.items()}