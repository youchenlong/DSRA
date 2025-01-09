REGISTRY = {}

from .basic_controller import BasicMAC
from .dsr_controller import DSRMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dsr_mac"] = DSRMAC
REGISTRY["maddpg_mac"] = MADDPGMAC