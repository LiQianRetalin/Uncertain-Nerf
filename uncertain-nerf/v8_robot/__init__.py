"""Robot-oriented V8 admission gates and integration contracts."""

from .gate import evaluate_candidate
from .pose_contract import PosePacket, parse_pose_packet

__all__ = ["PosePacket", "evaluate_candidate", "parse_pose_packet"]
