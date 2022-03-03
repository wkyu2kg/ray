import ray
from pprint import pprint
import socket
import time

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)

ray.init(address='auto')

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

# Create a placement group.
pg = placement_group([{"CPU": 2}, {"CPU": 2}], strategy="STRICT_SPREAD", name="default_spread")
ray.get(pg.ready())

pprint(placement_group_table(pg))
print(pg.bundle_specs)

@ray.remote(num_cpus=2, placement_group=pg)
def f1():
    print(socket.gethostname())
    time.sleep(1)
    return True

@ray.remote(num_cpus=2, placement_group=pg)
def f2():
    print(socket.gethostname())
    time.sleep(1)
    return True

a = f1.options(placement_group=pg).remote()
b = f2.options(placement_group=pg).remote()
ray.get([a, b])
