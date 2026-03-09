# Simulator for Checkpointing and Recovery: CPEN 533 (University of British Columbia)
"""
Metrics collection for checkpointing simulations.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EpochRecord:
    """One checkpoint interval (epoch) — either successful or ended by a failure."""
    start_time:           float
    end_time:             float
    had_failure:          bool
    work_done:            float   # useful computation completed this epoch
    wasted_work:          float   # computation lost to rollback
    recovery_time:        float   # time spent in failure recovery
    coordination_cost:    float   # time spent synchronising nodes before checkpoint
    checkpoint_cost:      float   # time spent writing the checkpoint


@dataclass
class Metrics:
    """Aggregated metrics over a full simulation run."""
    epochs: List[EpochRecord] = field(default_factory=list)

    def record_epoch(self, record: EpochRecord) -> None:
        self.epochs.append(record)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def total_time(self) -> float:
        return self.epochs[-1].end_time if self.epochs else 0.0

    @property
    def useful_work(self) -> float:
        return sum(e.work_done for e in self.epochs)

    @property
    def wasted_work(self) -> float:
        return sum(e.wasted_work for e in self.epochs)

    @property
    def total_recovery_time(self) -> float:
        return sum(e.recovery_time for e in self.epochs)

    @property
    def total_coordination_overhead(self) -> float:
        return sum(e.coordination_cost for e in self.epochs)

    @property
    def total_checkpoint_overhead(self) -> float:
        return sum(e.checkpoint_cost for e in self.epochs)

    @property
    def n_failures(self) -> int:
        return sum(1 for e in self.epochs if e.had_failure)

    @property
    def n_checkpoints(self) -> int:
        return sum(1 for e in self.epochs if not e.had_failure)

    @property
    def efficiency(self) -> float:
        """Fraction of total wall-clock time spent on useful computation."""
        return self.useful_work / self.total_time if self.total_time > 0 else 0.0

    def summary(self) -> dict:
        return {
            'efficiency':              self.efficiency,
            'useful_work':             self.useful_work,
            'wasted_work':             self.wasted_work,
            'recovery_time':           self.total_recovery_time,
            'coordination_overhead':   self.total_coordination_overhead,
            'checkpoint_overhead':     self.total_checkpoint_overhead,
            'n_failures':              self.n_failures,
            'n_checkpoints':           self.n_checkpoints,
            'total_time':              self.total_time,
        }
