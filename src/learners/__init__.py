from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .latent_q_learner import LatentQLearner
from .maddpg_learner import MADDPGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["latent_q_learner"] = LatentQLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
