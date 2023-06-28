import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, decimal_places=3):
        self._start_time = None
        self.decimal_places = decimal_places

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            self._start_time = None
            # raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        return round(elapsed_time, self.decimal_places)
