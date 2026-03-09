# Simulator for Checkpointing and Recovery: CPEN 533 (University of British Columbia)
"""
Coordinated checkpointing on n parallel nodes.

Key model assumptions
---------------------
* Failures are independent across nodes; any single failure causes all nodes
  to roll back to the most recent checkpoint.
* The *coordinator* drives time: it runs one checkpoint interval per iteration,
  sampling a failure time for every node at the start of each epoch.
* Checkpoint cost C is paid only on a *successful* epoch (no failure).
* Recovery time R (MTTR) is paid after a failure before work resumes.

Insight for students
--------------------
With n nodes the *system* MTBF is MTBF_node / n.
Larger n ⟹ more frequent failures ⟹ shorter optimal interval.
Daly's formula (see simulator.py) quantifies this trade-off.
"""
import simpy
from dataclasses import dataclass
from typing import Callable

from metrics import Metrics, EpochRecord


@dataclass
class SystemConfig:
    n_nodes:               int
    checkpoint_interval:   float     # T  — time between checkpoints
    checkpoint_cost:       float     # C  — time to write a checkpoint
    failure_sampler:       Callable  # () -> float, time-to-failure per node
    recovery_sampler:      Callable  # () -> float, recovery duration (MTTR)
    coordination_sampler:  Callable  # () -> float, pre-checkpoint coordination delay
    sim_duration:          float     # total wall-clock time to simulate

    def __post_init__(self):
        if self.checkpoint_cost >= self.checkpoint_interval:
            raise ValueError(
                f"checkpoint_cost ({self.checkpoint_cost}) must be less than "
                f"checkpoint_interval ({self.checkpoint_interval})."
            )


class ParallelSystem:
    """
    Simulates coordinated checkpointing with n_nodes parallel workers.

    Usage
    -----
    env = simpy.Environment()
    system = ParallelSystem(env, config)
    env.run(until=config.sim_duration)
    print(system.metrics.summary())
    """

    def __init__(self, env: simpy.Environment, config: SystemConfig):
        self.env = env
        self.config = config
        self.metrics = Metrics()
        self._process = env.process(self._coordinator())

    def _coordinator(self):
        """
        Main simulation loop.  Each iteration = one checkpoint epoch.

        Epoch timeline
        --------------
        Success:  |<--- T work --->|<- coord ->|<- C checkpoint ->|
        Failure:  |<- t_f work ->X  <--- R recovery ---|
                   ^last checkpoint                     ^restart from last ckpt

        The coordination delay models the overhead of synchronising all n nodes
        before the checkpoint write begins (e.g. barrier, message rounds).
        It is sampled fresh each successful epoch.
        """
        cfg = self.config

        while self.env.now < cfg.sim_duration:
            epoch_start = self.env.now

            # Draw an independent time-to-failure for every node.
            # The *system* fails at the minimum (first node to go down).
            node_ttfs = [cfg.failure_sampler() for _ in range(cfg.n_nodes)]
            t_fail = min(node_ttfs)   # time from epoch start to first failure

            if t_fail < cfg.checkpoint_interval:
                # ── FAILURE EPOCH ─────────────────────────────────────────
                # Run until the first node fails, then stall for recovery.
                yield self.env.timeout(t_fail)

                recovery = cfg.recovery_sampler()
                yield self.env.timeout(recovery)

                self.metrics.record_epoch(EpochRecord(
                    start_time=epoch_start,
                    end_time=self.env.now,
                    had_failure=True,
                    work_done=0.0,          # all work since last ckpt is lost
                    wasted_work=t_fail,
                    recovery_time=recovery,
                    coordination_cost=0.0,
                    checkpoint_cost=0.0,
                ))

            else:
                # ── SUCCESS EPOCH ─────────────────────────────────────────
                # No failure in [0, T]; coordinate, then write the checkpoint.
                yield self.env.timeout(cfg.checkpoint_interval)

                coord = cfg.coordination_sampler()
                yield self.env.timeout(coord)

                C = cfg.checkpoint_cost
                yield self.env.timeout(C)

                self.metrics.record_epoch(EpochRecord(
                    start_time=epoch_start,
                    end_time=self.env.now,
                    had_failure=False,
                    work_done=cfg.checkpoint_interval,
                    wasted_work=0.0,
                    recovery_time=0.0,
                    coordination_cost=coord,
                    checkpoint_cost=C,
                ))
