import ray
import ray.util.collective as collective
from pprint import pprint
import socket
import time

import numpy as np

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)

ray.init(namespace="coll", address="auto")

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

# Create a placement group.
pg = placement_group([{"CPU": 2}, {"CPU": 2}, {"CPU": 2}], strategy="STRICT_SPREAD", name="default_spread", lifetime=None)
ray.get(pg.ready())

pprint(placement_group_table(pg))
print(pg.bundle_specs)


@ray.remote(placement_group=pg)
class Worker:
   def __init__(self):
       self.send = np.ones((3, ), dtype=np.float32)
       self.recv = np.zeros((3, ), dtype=np.float32)

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       print(socket.gethostname())
       collective.allreduce(self.send, "default")
       return self.send

   def destroy(self):
       collective.destroy_collective_group()

num_workers = 3
workers = []
for i in range(num_workers):
   w = Worker.options(placement_group=pg).remote()
#   w = Worker.remote()
   workers.append(w)
_options = {
   "group_name": "default",
   "world_size": 3,
   "ranks": [0, 1, 2],
   "backend": "gloo"
}
collective.create_collective_group(workers, **_options)
results = ray.get([w.compute.remote() for w in workers])
results = ray.get([w.destroy.remote() for w in workers])

remove_placement_group(pg)
pprint(placement_group_table(pg))

ray.shutdown()
