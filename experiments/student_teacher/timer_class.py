# timer.py
# timer.py
import time
import sys
import resource
from functools import wraps, update_wrapper

def get_peak_memory_usage():
    """
    Returns a formatted string of the peak memory usage.
    On Linux, ru_maxrss is reported in kilobytes; on macOS it is in bytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':  # macOS: convert bytes to MB
        usage_mb = usage / (1024 ** 2)
    else:  # Linux: convert kilobytes to MB
        usage_mb = usage / 1024
    return f"Peak memory usage: {usage_mb:.2f} MB"

class Timer:
    def __init__(self, func=None, print_time=True, log_file=None, show_memory=False, text: str=None):
        """
        Initialize the Timer.
        
        :param func: Function to be timed (if used as a decorator).
        :param print_time: Whether to print the log messages.
        :param log_file: Optional path to a file for logging.
        :param show_memory: If True, includes memory usage in log messages.
        """
        self.func = func
        self.print_time = print_time
        self.log_file = log_file
        self.show_memory = show_memory
        self.text = text
        if func is not None:
            update_wrapper(self, func)

    def _log(self, message):
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        if self.print_time:
            print(message)

    def __call__(self, *args, **kwargs):
        if self.func is None:
            # Decorator called with arguments
            func = args[0]
            new_timer = Timer(
                func=func,
                print_time=self.print_time,
                log_file=self.log_file,
                show_memory=self.show_memory
            )
            update_wrapper(new_timer, func)
            return new_timer
        else:
            # Timing the function call
            start_time = time.perf_counter()
            result = self.func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Conditionally add memory info if enabled
            if self.show_memory:
                memory_info = get_peak_memory_usage()
                if self.text:
                    self._log(self.text)
                self._log(f"{self.func.__name__} took {execution_time:.4f} seconds. {memory_info}")
            else:
                self._log(f"{self.func.__name__} took {execution_time:.4f} seconds.")
            return result

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        execution_time = self.end_time - self.start_time
        
        # Conditionally add memory info if enabled
        if self.show_memory:
            memory_info = get_peak_memory_usage()
            if self.text:
                    self._log(self.text)
            self._log(f"Code block took {execution_time:.4f} seconds. {memory_info}")
        else:
            self._log(f"Code block took {execution_time:.4f} seconds.")


if __name__ == '__main__':
    # Example usage as a decorator with memory logging enabled
    @Timer(print_time=True, show_memory=True)
    def my_function(n):
        time.sleep(1)
        return n * 2

    output = my_function(5)
    print("Output:", output)

    # Example usage as a context manager with memory logging disabled
    with Timer(print_time=True, show_memory=False):
        time.sleep(1)
        result = 5 * 2
        print("Result:", result)
