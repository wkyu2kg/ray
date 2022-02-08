import socket
import time
import ray

#ray.init()
#ray.init(num_cpus=36, resources={'Custom': 3})
ray.init(address='auto')

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(36)]
print(ray.get(futures)) # [0, 1, 4, 9]

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        print(socket.gethostname())
        time.sleep(10)
        self.n += 1

    def read(self):
        return self.n

counters = [Counter.remote() for i in range(36)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures)) # [1, 1, 1, 1]

ray.shutdown()
