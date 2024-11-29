import json
import numpy as np
import torch as th


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


def save_eval_data(data):
    rewards, _, _ = zip(*data)

    print(np.mean([np.sum(r) for r in rewards]))

    to_write = {
        f"ep_{i}": [
            {
                "reward": r,
                "obs": s,
                "action": a,
            }
            for r, s, a in zip(*episode)
        ]
        for i, episode in enumerate(data)
    }

    with open("evaluation.json", "w") as f:
        import json

        json.dump(to_write, f, cls=Serializer, indent=2)
