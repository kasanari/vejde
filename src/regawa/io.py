
from regawa import GroundValue, GroundingValueType


def obs_to_json_friendly_obs(obs: dict[GroundValue, GroundingValueType]) -> dict[str, GroundingValueType]:
	return {"__".join(k): v for k, v in obs.items()}

def json_friendly_obs_to_obs(obs: dict[str, GroundingValueType]) -> dict[GroundValue, GroundingValueType]:
	return {tuple(k.split("__")): v for k, v in obs.items()}