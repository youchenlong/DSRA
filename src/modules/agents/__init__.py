REGISTRY = {}

from .rnn_agent import RNNAgent
from .dsr_agent import DSRAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dsr"] = DSRAgent