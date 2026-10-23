"""Frozen user-approved sampling contract; speed changes must preserve it."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class OriginalContract:
    arm_radius_rad: float = .15
    hand_limit_margin: float = .1
    knot_interval_seconds: float = 1.
    extra_hold_probability: float = 0.
    fps: int = 24
    physics_hz: int = 120
    warmup_steps: int = 240
    arm_stiffness: float = 1000.
    arm_damping: float = 80.
    hand_stiffness: float = 3.
    hand_damping: float = .1
    position_iterations: int = 16
    velocity_iterations: int = 4

CONTRACT=OriginalContract()

def original_commands(seed,center,limits,arm,seconds=5):
    assert isinstance(seconds,int) and seconds>=1
    lo,hi=np.asarray(limits,dtype=np.float64).T.copy();center=np.asarray(center,dtype=np.float64);arm=np.asarray(arm,dtype=bool)
    lo[arm]=np.maximum(lo[arm],center[arm]-CONTRACT.arm_radius_rad)
    hi[arm]=np.minimum(hi[arm],center[arm]+CONTRACT.arm_radius_rad)
    span=hi-lo;lo[~arm]+=CONTRACT.hand_limit_margin*span[~arm];hi[~arm]-=CONTRACT.hand_limit_margin*span[~arm]
    initial=np.clip(center,lo,hi)
    knots=np.random.default_rng(seed).uniform(lo,hi,(seconds+1,len(center)));knots[0]=initial
    phase=np.arange(1,seconds*CONTRACT.fps+1)/CONTRACT.fps
    index=np.minimum(phase.astype(np.int64),seconds-1);fraction=phase-index
    blend=fraction*fraction*(3-2*fraction)
    actions=knots[index]*(1-blend[:,None])+knots[index+1]*blend[:,None]
    return initial,knots,actions

def metadata():return asdict(CONTRACT)
