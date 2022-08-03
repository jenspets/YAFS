#!/usr/bin/env python3

import networkx as nx
import time
from pathlib import Path
import logging.config
import random
import json
import matplotlib.pyplot as plt
import pprint
import argparse
import itertools

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology
from yafs.placement import Placement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.volatility import Volatility, ExponentialVolatility
from yafs.distribution import deterministic_distribution
from yafs.stats import Stats

DEFAULT_SIZE = 100   # Default number of nodes in the network
NSOURCES     = .1    # Percentage of nodes that will act as message sources
PSERVER      = .2    # Percentage of ellegible nodes that are selected as a server

class FogRandomStaticPlacement(Placement):
    '''
    The class places services at random nodes that have minimum required MEM configuration. 
    '''

    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        pserver = PSERVER
        pots = []
        minmem = 1E3    # Min 1 GB MEM

        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            if mem > minmem and random.random() <= pserver:
                pots.append(n)
        
        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], pots[0:1])


class FogTreeSiblingConnectionPlacement(Placement):
    '''
    The class generate a tree structure from the network, with connections between siblings. 
    The service is placed in all nodes that have minimum requirements, and that is not a leaf node
    '''
    original_G = None
    
    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        minmem = 1E3
        pserver = PSERVER
        serverlist = []

        # Select nodes to act as servers
        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            if mem > minmem and sim.topology.G.degree(n) > 1 and random.random() <= pserver:
                serverlist.append(n)

        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], serverlist)


def deploy_random_lowresource_sources(sim):
    '''
    Deploy a number of sources, at the most memory constrained nodes, each with a message sending distribution. 
    '''
    nsources = int(nx.number_of_nodes(sim.topology.G) * NSOURCES)
    # nsources = 10

    # make a list of sources, select the nsources with lowest MEM
    psources = [x for x in sorted(sim.topology.G.nodes(), key=lambda x: sim.topology.G.nodes()[x]['MEM'])][:nsources]

    for a in sim.apps:
        for i in psources:
            # Select a random message from app
            msg = sim.apps[a].get_message(random.sample(sorted(sim.apps[a].messages), 1)[0])
            dist = deterministic_distribution(100, name='Deterministic')
            idDES = sim.deploy_source(a, id_node=i, msg=msg, distribution=dist)


def deploy_leaf_nodes_sources(sim):
    '''
    Deploy sources at leaf nodes in a tree, which means nodes without any descendants. This needs the Depth attribute in nodes
    '''
    nsources = int(nx.number_of_nodes(sim.topology.G) * NSOURCES)
    deg = 1
    psources = [x for x in sim.topology.G.nodes() if sim.topology.G.degree(x) == deg]
    maxdeg = max(nx.degree(sim.topology.G), key=lambda x: x[1])[1]
    while len(psources) < nsources:
        print(f'Too few sources with degree {deg}, increasing maximum degree. len(psources) = {len(psources)}/{nsources}')
        deg += 1 
        psources = [x for x in sim.topology.G.nodes() if sim.topology.G.degree(x) <= deg]
        if deg > maxdeg:
            print(f'Reached max degree without finding enough nodes: {maxdeg}')
            break
    random.shuffle(psources)
    psources = psources[:nsources]
    
    
    for a in sim.apps:
        for i in psources:
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


def add_sibling_edge(graph, parent, skip, p, pr, bw, d):
    children = nx.descendants_at_distance(graph, parent, 1)
    children = set(children) - skip
    skip.update(children)
    children = list(children)
    nx.set_node_attributes(graph, {parent: d}, name='Depth')
    
    # print(' '*d + f'{parent} - {children} - {skip}')
    if len(children) < 1:
        return

    if len(children) > 1:
        pairs = []
        for pair in itertools.combinations(children, 2):
            if p <= random.random():
                pairs.append(pair)
                graph.add_edges_from(pairs)
                attrs = {pair: {'PR': random.gammavariate(*pr), 'BW': random.gammavariate(*bw)} for pair in pairs}
                nx.set_edge_attributes(graph, attrs)

    for child in children:
        add_sibling_edge(graph, child, skip, p, pr, bw, d+1)

def set_tree(topology, p, pr, bw):
    original_G = topology.G
    stg = nx.maximum_spanning_tree(topology.G, weight='BW')
    for n in stg.nodes():
        print(f'Node: {n}: {stg.nodes()[n]}')
    for e in stg.edges():
        print(f'Edge: {e}: {stg.edges()[e]}')
    # find center, and add connection between children, using the supplied parameters for the gamma distribution
    center = nx.center(stg)[0]
    add_sibling_edge(stg, center, set([center]), p, pr, bw, 0)
    
    topology.G = stg
    return original_G, stg

def main(stop_time, graphgen, serviceplacement, sourcedeployment, it, folder_results, folder_data, create_tree):
    # Topology
    
    t = Topology()
    t.G = graphgen['f'](**graphgen['args'])
    # t.G = nx.generators.random_internet_as_graph(100)

    fe = open(f'{folder_results}/{it:04}_edges.csv', 'w')
    fn = open(f'{folder_results}/{it:04}_nodes.csv', 'w')
    
    # Set the attributes of the edges and nodes
    # attrPR_BW = {x: 1 for x in t.G.edges()}
    # nx.set_edge_attributes(t.G, name='PR', values=attrPR_BW)
    # nx.set_edge_attributes(t.G, name='BW', values=attrPR_BW)
    
    # Latency for edge is: message_size/(BW * 1E6) + PR
    attrPR = {x: random.gammavariate(1, .2) for x in t.G.edges} # Gives mostly valuse between 0 and 1 second
    attrBW = {x: random.gammavariate(1.5, 1) for x in t.G.edges}  # BW given in Mb/s
    nx.set_edge_attributes(t.G, name='PR', values=attrPR)
    nx.set_edge_attributes(t.G, name='BW', values=attrBW)
    print('PR and BW')
    fe.write('Edge;PR;BW\n')
    for e in t.G.edges():
        print(f'e: {e}, PR: {t.G.edges()[e]["PR"]} BW: {t.G.edges()[e]["BW"]}')
        fe.write(f'{e};{t.G.edges()[e]["PR"]};{t.G.edges()[e]["BW"]}\n')
    fe.close()
        
    attrIPT = {x: random.gauss(1000, 200) for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='IPT', values=attrIPT)
    
    # MEM (in MB) set at random from a n.e.d. 1E-4 gives very few at 100G, avg at 10G
    attrMEM = {x: int(random.expovariate(1E-4)) for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='MEM', values=attrMEM)
    #plt.hist([attrMEM[x] for x in attrMEM])
    #plt.show()
    #pprint.pprint(attrMEM)
    print('IPT and MEM')
    fn.write('Node;IPT;MEM\n')
    for n in t.G.nodes():
        print(f'n: {n}, IPT: {t.G.nodes()[n]["IPT"]}, MEM: {t.G.nodes()[n]["MEM"]}')
        fn.write(f'{n};{t.G.nodes()[n]["IPT"]};{t.G.nodes()[n]["MEM"]}\n')
    fn.close()

    # Create a maximum spanning tree from the graph, and exchange the graph with the subgraph
    nx.write_gexf(t.G, f'{folder_results}/{it:04}_original.gexf')
    original_G = set_tree(t, .5, (1, .2), (1.5, 1))
    
    nx.write_gexf(t.G, f'{folder_results}/{it:04}_spanning_tree.gexf')
    with open(f'{folder_results}/{it:04}_tree_edges.csv', 'w') as f:
        f.write('Edge;PR;BW\n')
        for e in t.G.edges():
            f.write(f'{e};{t.G.edges()[e]["PR"]};{t.G.edges()[e]["BW"]}\n')

    # Application
    japp = json.load(open(f'{folder_data}/appDef.json'))
    apps = create_applications_from_json(japp)

    # Service placement
    # placement = FogRandomStaticPlacement(name='Placement')
    placement = serviceplacement(name='Placement')
    
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
    s = Sim(t, default_results_path=f'{folder_results}/{it:04}_sim_trace')

    # Deploy services
    for a in apps:
        s.deploy_app_vol(apps[a], placement, routing, volatility[a])

    # Deploy sources
    sourcedeployment(s)

    # Run simulation
    logging.info(f' Performing simulation {it}')
    s.run(stop_time)
    # s.print_debug_assignaments()

    stats = Stats(defaultPath=f'{folder_results}/{it:04}_sim_trace')
    stats.showVolatility(stop_time, t)


if "__main__" == __name__:
    graphgen = {'ba': {'f': nx.generators.barabasi_albert_graph,
                       'name': 'Barabassi-Albert',
                       'args': {'n': DEFAULT_SIZE, 'm': 2}},
                'er': {'f': nx.generators.erdos_renyi_graph,
                       'name': 'Erdos-Renyi',
                       'args': {'n': DEFAULT_SIZE, 'p': 0.05}},
                'ws': {'f': nx.generators.connected_watts_strogatz_graph,
                       'name': 'Connected Watts-Strogatz',
                       'args': {'n': DEFAULT_SIZE, 'k': 5, 'p': .05}},
                'rl': {'f': nx.generators.random_lobster,
                       'name': 'Random Lobster',
                       'args': {'n': DEFAULT_SIZE, 'p1': .05, 'p2': .01}}}

    placement = {'randomstatic': FogRandomStaticPlacement,
                 'treesibling': FogTreeSiblingConnectionPlacement}
    
    sourcedeployment = {'lowresource': deploy_random_lowresource_sources,
                        'leafnodes': deploy_leaf_nodes_sources}
    
    aparse = argparse.ArgumentParser(description='Create a network, test various fog network structures')
    aparse.add_argument('-r', '--results', default='results', help='Directory to store results')
    aparse.add_argument('-d', '--datadir', default='data2', help='Directory for app and logging setup')
    aparse.add_argument('-g', '--graph', choices=graphgen.keys(), default='ba', help='Graph type to use for network creation')
    aparse.add_argument('-a', '--args', help='Named arguments for graph generator')
    aparse.add_argument('-p', '--placement', choices=placement.keys(), default='randomstatic', help='Fog service placement algorithm')
    aparse.add_argument('-s', '--source', choices=sourcedeployment.keys(), default='lowresource', help='Message source placement')
    aparse.add_argument('-t', '--tree', action='store_true', help='Create a maximum spanning tree w/ sibling connections from the generated graph')
    args = aparse.parse_args()

    sim_duration = 600
    iterations = 1

    folder_results = Path(args.results)
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)

    # logging setup
    folder_data = args.datadir
    logging.config.fileConfig(f'{folder_data}/logging.ini')

    # Set graph algorithm parameters
    if args.args:
        a = json.loads(args.args)
        for e in a:
            graphgen[args.graph]['args'][e] = a[e]

    print(f'Parameters: Graph generator   = {graphgen[args.graph]["name"]}, args = {graphgen[args.graph]["args"]}')
    print(f'            Placement         = {args.placement}')
    print(f'            Source deployment = {args.source}')
    gcpy = graphgen[args.graph].copy()
    pcpy = placement.copy()
    scpy = sourcedeployment.copy()
    
    settings = {'graph': {args.graph: gcpy},
                'placement': {args.placement: pcpy[args.placement]},
                'sourcedeployment': {args.source: scpy[args.source]},
                'datadir': args.datadir}
    # change pointers to strings
    settings['graph'][args.graph]['f'] = str(settings['graph'][args.graph]['f'])
    settings['placement'][args.placement] = str(settings['placement'][args.placement])
    settings['sourcedeployment'][args.source] = str(settings['sourcedeployment'][args.source])
    with open(f'{args.results}/settings.json', 'w') as f:
        json.dump(settings, f, indent=4)
    
    for i in range(iterations):
        tstart = time.time()
        main(sim_duration,
             graphgen[args.graph],
             placement[args.placement],
             sourcedeployment[args.source],
             i,
             folder_results,
             folder_data,
             args.tree)
        print(f'\n--- Iteration: {i}: {time.time() - tstart} seconds ---')

    print('simulation finished')
