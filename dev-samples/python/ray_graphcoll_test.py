import os
import datasets as ds
import numpy as np
import pytest
import ray
import ray.util.collective as collective

import katana.distributed
from katana.distributed import Graph

from katana.local import DynamicBitset

ray.init(namespace='coll', address='auto', _redis_password='5241590000000000', runtime_env={"working_dir": "./"})

@ray.remote
class Worker:
   def __init__(self):
       import katana.distributed

       katana.distributed.initialize()
       self.send = np.ones((4, ), dtype=np.float32)
       self.recv = np.zeros((4, ), dtype=np.float32)
       # two statements below are equivalent. use 1st if ds does not load
       # self.graph = Graph("gs://katana-demo-datasets/unit-test-inputs/gnn/tester")
       self.graph = ds.rdg_dataset_url("gnn_tester", "local")

   def setup(self, world_size, rank):
       collective.init_collective_group(world_size, rank, "gloo", "default")
       return True

   def compute(self):
       collective.allreduce(self.send, "default")
       return self.send

   def destroy(self):
       collective.destroy_collective_group()

   def graph_vector(self, datasource):
       tester_graph = self.graph
       bitset = DynamicBitset(len(tester_graph))

       dimensions = 3
       data_to_sync = np.empty((len(tester_graph) * dimensions), dtype=np.uint32)
       data_to_sync[:] = 0

       # set only on masters: set the user id
       for master_lid in tester_graph.masters:
           for i in range(dimensions):
               data_to_sync[master_lid * dimensions + i] = tester_graph.local_to_user_id(master_lid)
       # set entire bitset to true
       for i in range(len(tester_graph)):
           bitset[i] = True

       # before sync, make sure mirrors are empty
       for mirror_lid in tester_graph.mirrors:
           for i in range(dimensions):
       	       assert data_to_sync[mirror_lid * dimensions + i] == 0

       # XXX: use <<Ray collectives>> for synchronization
       collective.allreduce(self.send, "default")
       
       # all rows should have the correct user id after this sync
       for lid in range(tester_graph.num_nodes()):
           for i in range(dimensions):
       	       assert data_to_sync[lid * dimensions + i] == tester_graph.local_to_user_id(lid)

       # set all masters to 0
       for master_lid in tester_graph.masters:
           for i in range(dimensions):
               data_to_sync[master_lid * dimensions + i] = 0

           # only even nodes will get sync'd
           if tester_graph.local_to_user_id(master_lid) % 2 == 0:
               bitset[master_lid] = True

       # do a "before sync" check to make sure values are different
       # before and after the sync call
       for mirror_lid in tester_graph.mirrors:
           for i in range(dimensions):
               assert data_to_sync[mirror_lid * dimensions + i] == tester_graph.local_to_user_id(mirror_lid)

       # XXX: use <<Ray collectives>> for synchronization
       collective.allreduce(self.send, "default")

       # mirrors with even user ids will be changed; odds remain the same
       for mirror_lid in tester_graph.mirrors:
           if tester_graph.local_to_user_id(mirror_lid) % 2 == 0:
               for i in range(dimensions):
                   assert data_to_sync[mirror_lid * dimensions + i] == 0
           else:
               for i in range(dimensions):
                   assert data_to_sync[mirror_lid * dimensions + i] == tester_graph.local_to_user_id(mirror_lid)

       obj = ray.put(data_to_sync)
       return obj

# imperative
num_workers = 2
workers = []

def test_gluon_vector_comm(datasource):
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
    for w in workers:
        w.graph_vector.remote(datasource)

    # XXX: invoke additional out-of-band tasks for synchronization
    results = ray.get([w.compute.remote() for w in workers])
    results = ray.get([w.destroy.remote() for w in workers])


ray.shutdown()
