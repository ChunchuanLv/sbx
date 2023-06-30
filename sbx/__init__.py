import os

from sbx.dqn import DQN
from sbx.droq import DroQ
from sbx.ppo import PPO
from sbx.sac import SAC
from sbx.hsac import HSAC
from sbx.tqc import TQC
from sbx.hppo import HPPO
from sbx.eodppo import EODPPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "DQN",
    "DroQ",
    "PPO",
    "HPPO",
    "EODPPO",
    "SAC",
    "HSAC",
    "TQC",
]
