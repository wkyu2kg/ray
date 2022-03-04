import os
import datasets as ds
import numpy as np
import pytest
import ray
import ray.util.collective as collective
from pprint import pprint

import socket
import time

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)


import katana.distributed
from katana.distributed import Graph

@ray.remote
class Worker:
   def __init__(self):
       import katana.distributed

       katana.distributed.initialize()

       self.send = np.ones((3, ), dtype=np.float32)
       self.recv = np.zeros((3, ), dtype=np.float32)
       self.graph = Graph("gs://katana-demo-datasets/unit-test-inputs/gnn/tester")

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       print(socket.gethostname())
       collective.allreduce(self.send, "default")
       return self.send

   def destroy(self):
       collective.destroy_group()

def test_gluon_vector_comm():
    ctx = ray.init(namespace="gluon", address="auto")
    print(ray.get_runtime_context().namespace)

    num_workers = 3
    workers = []
    init_rets = []

    # Create a placement group.
    pg = placement_group([{"CPU": 2}, {"CPU": 2}, {"CPU": 2}], strategy="STRICT_SPREAD", name="default_spread", lifetime=None)
    ray.get(pg.ready())

    pprint(placement_group_table(pg))
    print(pg.bundle_specs)

    for i in range(num_workers):
       w = Worker.options(placement_group=pg).remote()
       workers.append(w)
    _options = {
       "group_name": "default",
       "world_size": 3,
       "ranks": [0, 1, 2],
       "backend": "gloo"
    }
    collective.create_collective_group(workers, **_options)
    results = ray.get([w.compute.remote() for w in workers])

    remove_placement_group(pg)
    pprint(placement_group_table(pg))
    ray.shutdown()
