#!/usr/bin/env python3

import networkx as nx
import time
from pathlib import Path
import logging.config
import random
import json
import matplotlib.pyplot as plt
import pprint

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology
from yafs.placement import Placement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.volatility import Volatility, ExponentialVolatility
from yafs.distribution import deterministic_distribution
from yafs.stats import Stats

class FogPlacement(Placement):
    ''' 
    The class will place services at the nodes that have a certain MEM configuration
    '''

    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        pserver = .2
        pots = []
        minmem = 1E3    # Min 1 GB mem 
        
        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            if mem > minmem and random.random() <= pserver:
                pots.append(n)
        
        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], pots)


def deploy_sources(sim):
    '''
    Deploy a number of sources, each with a distribution 
    '''
    nsources = 10

    # make a list of sources, select the nsources with lowest MEM
    psources = [x for x in sorted(sim.topology.G.nodes(), key=lambda x: sim.topology.G.nodes()[x]['MEM'])][:nsources]

    for a in sim.apps:
        for i in psources:
            # Select a random message from app
            msg = sim.apps[a].get_message(random.sample(sorted(sim.apps[a].messages), 1)[0])
            dist = deterministic_distribution(100, name='Deterministic')
            idDES = sim.deploy_source(a, id_node=i, msg=msg, distribution=dist)


def get_eraselimits_gcbased(mem, writing_rate=120, scale=10):
    ''' Calculate erase time volatility limits for a uniform distribution from the MeM size and the writing rate (s/w).
    This should also take into account the dynamic load of the node, as much writing faster will overwrite obsolete data.
    For now, assume a static load on the nodes, set individually. I don't think YAFS records the load in each node.
    The derivative should be decreasing with higher number, but not be negative or zero.
    '''

    nmin = .5  # Min/max number of remaining sectors after a GC, based loosely on Contiki volatility paper

    # tmax = writing_rate * mem * scale

    mean = (mem * scale) / writing_rate
    tmax = random.gauss(mean, mean/10)
    tmin = nmin*tmax
    
    return (tmin, tmax)


def main(stop_time, it, folder_results):
    # Topology
    t = Topology()
    t.G  = nx.generators.barabasi_albert_graph(100, 2)
    # t.G = nx.generators.random_internet_as_graph(100)

    # Set the attributes of the edges and nodes
    attrPR_BW = {x: 1 for x in t.G.edges()}
    nx.set_edge_attributes(t.G, name='PR', values=attrPR_BW)
    nx.set_edge_attributes(t.G, name='BW', values=attrPR_BW)

    attrIPT = {x: 100 for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='IPT', values=attrIPT)
    # MEM (in MB) set at random from a n.e.d. 1E-4 gives very few at 100G, avg at 10G
    attrMEM = {x: int(random.expovariate(1E-4)) for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='MEM', values=attrMEM)
    #plt.hist([attrMEM[x] for x in attrMEM])
    #plt.show()
    #pprint.pprint(attrMEM)

    # Application
    japp = json.load(open('data/appDef.json'))
    apps = create_applications_from_json(japp)

    # Service placement
    placement = FogPlacement(name='Placement')

    # Routing algorithm
    routing = DeviceSpeedAwareRouting()

    # Volatility
    volatility = {}
    for a in apps:
        volatility[a] = ExponentialVolatility(a, t)
        volatility[a].set_erasedistr(1/60, vtype=Volatility.SOURCE)
        volatility[a].set_erasedistr(1/10, vtype=Volatility.PROXY)
        volatility[a].set_erasedistr(1/600, vtype=Volatility.SINK)
        volatility[a].set_unlinkdistr(1/300, vtype=Volatility.SOURCE)
        volatility[a].set_unlinkdistr(1/300, vtype=Volatility.PROXY)
        volatility[a].set_unlinkdistr(1/300, vtype=Volatility.SINK)
    
    # Simulation
    s = Sim(t, default_results_path=folder_results+'sim_trace')

    # Deploy services
    for a in apps:
        s.deploy_app_vol(apps[a], placement, routing, volatility[a])

    # Deploy sources
    deploy_sources(s)

    # Run simulation
    logging.info(f' Performing simulation {it}')
    s.run(stop_time)
    # s.print_debug_assignaments()

    stats = Stats(defaultPath=folder_results+'sim_trace')
    stats.showVolatility(stop_time, t)


if "__main__" == __name__:
    sim_duration = 600
    iterations = 1

    folder_results = Path('results')
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results) + '/'

        # logging setup
    logging.config.fileConfig('data/logging.ini')
    
    for i in range(iterations):
        tstart = time.time()
        main(sim_duration, i, folder_results)
        print(f'\n--- Iteration: {i}: {time.time() - tstart} seconds ---')

    print('simulation finished')
