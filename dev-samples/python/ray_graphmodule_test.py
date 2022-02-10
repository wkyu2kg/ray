import datasets as ds
import numpy as np
import pytest
import ray
import ray.util.collective as collective

import katana.distributed
from katana.distributed import Graph

ray.init(namespace='coll', address='auto', _redis_password='5241590000000000', runtime_env={"env_vars": {"PYTHONPATH": "/home/wkyu/katana/master/external/katana/katana_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/external/katana/katana_python_build/python:/home/wkyu/katana/master/katana_enterprise_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/katana_enterprise_python_build/python:/home/wkyu/katana/master/external/katana/katana_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/external/katana/katana_python_build/python:/home/wkyu/katana/katana-enterprise-master/python/test"}})

@ray.remote
class Worker:
   def __init__(self):
       katana.distributed.initialize()

       self.send = np.ones((4, ), dtype=np.float32)
       self.recv = np.zeros((4, ), dtype=np.float32)
       self.graph = Graph("gs://katana-demo-datasets/unit-test-inputs/gnn/tester")

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       collective.allreduce(self.send, "default")
       return self.send

num_workers = 2
workers = []
init_rets = []
tester_graph = ds.rdg_dataset_url("gnn_tester", "local")

def test_gluon_vector_comm():
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

ray.shutdown()
