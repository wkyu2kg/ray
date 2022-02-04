import ray
import time

ray.init()
#  ray.init(num_cpus=8, num_gpus=4, resources={'Custom': 2})


# A regular Python function.
def my_function():
    return 1

# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1

# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ``ray.get``.
assert ray.get(obj_ref) == 1

@ray.remote
def slow_function():
    time.sleep(10)
    return 1

# Invocations of Ray remote functions happen in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
for _ in range(4):
    # This doesn't block.
    slow_function.remote()

@ray.remote
def function_with_an_argument(value):
    return value + 1


obj_ref1 = my_function.remote()
assert ray.get(obj_ref1) == 1

# You can pass an object ref as an argument to another Ray remote function.
obj_ref2 = function_with_an_argument.remote(obj_ref1)
assert ray.get(obj_ref2) == 2

@ray.remote(num_returns=3)
def return_multiple():
    return 1, 2, 3

a, b, c = return_multiple.remote()

@ray.remote
def blocking_operation():
    time.sleep(10e6)

obj_ref = blocking_operation.remote()
ray.cancel(obj_ref)

from ray.exceptions import TaskCancelledError

try:
    ray.get(obj_ref)
except TaskCancelledError:
    print("Object reference was cancelled.")

# Get the value of one object ref.
obj_ref = ray.put(1)
assert ray.get(obj_ref) == 1

# Get the values of multiple object refs in parallel.
assert ray.get([ray.put(i) for i in range(3)]) == [0, 1, 2]

# You can also set a timeout to return early from a ``get`` that's blocking for too long.
from ray.exceptions import GetTimeoutError

@ray.remote
def long_running_function():
    time.sleep(8)

obj_ref = long_running_function.remote()
try:
    ray.get(obj_ref, timeout=4)
except GetTimeoutError:
    print("`get` timed out.")

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

# Create an actor from this class.
counter = Counter.remote()

# Specify required resources for an actor.
@ray.remote(num_cpus=2, num_gpus=0.5)
class Actor(object):
    pass

# Call the actor.
obj_ref = counter.increment.remote()
assert ray.get(obj_ref) == 1

# Create ten Counter actors.
counters = [Counter.remote() for _ in range(10)]

# Increment each Counter once and get the results. These tasks all happen in
# parallel.
results = ray.get([c.increment.remote() for c in counters])
print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Increment the first Counter five times. These tasks are executed serially
# and share state.
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)  # prints [2, 3, 4, 5, 6]

ray.shutdown()
