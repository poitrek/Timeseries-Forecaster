import time


class Timer:

    def start_measure_time(self):
        self.start_time = time.perf_counter()

    def stop_measure_time(self):
        self.stop_time = time.perf_counter()
        self.time_elapsed = self.stop_time - self.start_time
