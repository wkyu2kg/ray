import ray
import ray.util.collective as collective

import numpy as np

ray.init(namespace="coll", address="auto")

@ray.remote
class Worker:
   def __init__(self):
       self.send = np.ones((4, ), dtype=np.float32)
       self.recv = np.zeros((4, ), dtype=np.float32)

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       collective.allreduce(self.send, "default")
       return self.send

   def destroy(self):
       collective.destroy_group()

num_workers = 2
workers = []
init_rets = []

# declarative
for i in range(num_workers):
   w = Worker.remote()
   workers.append(w)
_options = {
   "group_name": "default",
   "world_size": 2,
   "ranks": [0, 1],
   "backend": "gloo"
}
collective.create_collective_group(workers, **_options)
results = ray.get([w.compute.remote() for w in workers])
