import simpy

class Task:
    """Simulates a simple task with start/stop primitives and time measurements."""

    def __init__(self, env: simpy.Environment, delay: float = 0):
        self.env = env
        self.delay = delay
        self.running_time = 0.0
        self._task = None

    def start(self):
        """Start the process."""
        self._task = self.env.process(self.run())

    def stop(self):
        """Interrupt the process."""
        if self._task and self._task.is_alive:
            self._task.interrupt("Stop")

    def action(self):
        yield self.env.timeout(self.delay)

    def run(self):
        while True:
            yield self.env.process(self.action())