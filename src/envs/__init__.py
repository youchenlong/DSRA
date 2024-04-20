from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .stag_hunt import StagHunt
# from .matrix_game.nstep_matrix_game import NStepMatrixGame
# from .rware import RWAREEnv
from .lbforaging import ForagingEnv
# from .sisl import SislEnv
# from .mpe import MPEEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
# REGISTRY["nstep_matrix"] = partial(env_fn, env=NStepMatrixGame)
# REGISTRY["rware"] = partial(env_fn, env=RWAREEnv)
REGISTRY["foraging"] = partial(env_fn, env=ForagingEnv)
# REGISTRY["sisl"] = partial(env_fn, env=SislEnv)
# REGISTRY["mpe"] = partial(env_fn, env=MPEEnv)
if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
