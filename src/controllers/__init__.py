REGISTRY = {}

from .basic_controller import BasicMAC
from .dsr_controller import DSRMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["dsr_mac"] = DSRMAC