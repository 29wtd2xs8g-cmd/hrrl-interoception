"""Configuration objects. Every experimental knob lives here so runs are reproducible
from the saved config alone."""
from dataclasses import dataclass, field, asdict


@dataclass
class EnvConfig:
    size: int = 7
    # (row, col, internal variable fed by this resource). Var 0 = energy, var 1 = water.
    # Placed in opposite corners so that satisfying one need costs the other.
    resources: tuple = ((0, 0, 0), (6, 6, 1))
    n_vars: int = 2
    setpoint: float = 10.0
    h_max: float = 20.0           # physiological ceiling (gut capacity)
    init_low: float = 6.0
    init_high: float = 10.0
    bite: float = 1.0             # amount ingested per consume action
    basal_cost: tuple = (0.04, 0.04)   # metabolic loss per step, per variable
    move_cost: tuple = (0.04, 0.02)    # extra loss when attempting to move
    max_steps: int = 1000         # training episode length (truncation, not death)

    # Temporal variant: announced scarcity, during which consuming yields nothing.
    scarcity: bool = False
    scarcity_warning: int = 60    # steps between the cue and the onset
    scarcity_duration: int = 250  # 250 * basal 0.04 = 10 = setpoint: lethal without storage
    scarcity_interval: tuple = (300, 600)  # gap between scarcity episodes
    scarcity_cue: bool = True     # False: the warning is hidden (control for anticipation)


@dataclass
class RewardConfig:
    condition: str = "hrrl"       # "hrrl" | "maximizer" | "saturating"
    drive_n: float = 4.0          # Keramati & Gutkin (2014): D(H) = (sum |h*_i - h_i|^n)^(1/m)
    drive_m: float = 3.0
    sigma: float = 0.0            # interoceptive noise, applied to observation AND reward
    sat_window: int = 50          # saturating control: sliding window (steps)
    sat_cap: tuple = None         # bites rewarded per window per variable; None = matched to metabolic need
    # Same terminal penalty in every condition, so all agents value survival and only
    # the satiation signal differs. ~ drive at death (one variable at 0): (10^4)^(1/3) = 21.5.
    # Set to 0 for pure HRRL, where death only costs the lost future drive reduction.
    death_penalty: float = 20.0


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    hidden: int = 128
    buffer_size: int = 100_000
    batch_size: int = 64
    train_every: int = 4
    target_update: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    learning_starts: int = 2_000
    grad_clip: float = 10.0


@dataclass
class RunConfig:
    seed: int = 0
    train_steps: int = 300_000
    eval_steps: int = 5_000
    eval_eps: float = 0.02
    log_every: int = 1_000
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self):
        return asdict(self)
