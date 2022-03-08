import os
import datasets as ds
import numpy as np
import pytest
import ray
import ray.util.collective as collective

import katana.distributed
from katana.distributed import Graph

# This statement does not work because Graph is a name 
# for both a local variable and a module

#ray.init(namespace='coll', address='auto', _redis_password='5241590000000000', runtime_env={"py_modules": [Graph], "working_dir": "./"})

# Either of the two statements below can work.

#ray.init(namespace='coll', address='auto', _redis_password='5241590000000000', runtime_env={"env_vars": {"PYTHONPATH": "/home/wkyu/katana/master/external/katana/katana_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/external/katana/katana_python_build/python:/home/wkyu/katana/master/katana_enterprise_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/katana_enterprise_python_build/python:/home/wkyu/katana/master/external/katana/katana_python_build/build/lib.linux-x86_64-3.8:/home/wkyu/katana/master/external/katana/katana_python_build/python:/home/wkyu/katana/katana-enterprise-master/python/test"}})
ray.init(namespace='coll', address='auto', _redis_password='5241590000000000', runtime_env={"working_dir": "./"})

@ray.remote
class Worker:
   def __init__(self):
       # N.B.: Need to explicitly import a module in the worker since 02/27/22
       import katana.distributed

       katana.distributed.initialize()

       self.send = np.ones((4, ), dtype=np.float32)
       self.recv = np.zeros((4, ), dtype=np.float32)
       print(os.environ["PYTHONPATH"])
       self.graph = Graph("gs://katana-demo-datasets/unit-test-inputs/gnn/tester")
       self.graph = ds.rdg_dataset_url("gnn_tester", "local")

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       collective.allreduce(self.send, "default")
       return self.send

   def destroy(self):
       collective.destroy_collective_group()


num_workers = 2
workers = []
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
    results = ray.get([w.destroy.remote() for w in workers])

ray.shutdown()
