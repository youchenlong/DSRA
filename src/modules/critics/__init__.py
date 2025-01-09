from .coma import COMACritic
from .maddpg import MADDPGCritic

REGISTRY = {}
REGISTRY["coma_critic"] = COMACritic
REGISTRY["maddpg_critic"] = MADDPGCritic