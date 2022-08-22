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
import scipy.stats as spstats

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
MINMEM       = 1E3   # Minimum memory for a fog server, in MB
PR_gv_alpha  = 1     # Propagation delay, gammavariate settings
PR_gv_beta   = .2    # Propagation delay, gammavariate settings
BW_gv_alpha  = 1.5   # Bandwidth, gammavariate settings
BW_gv_beta   = 1     # Bandwidth, gammavariate settings
PRC_gv_alpha = PR_gv_alpha * 10  # A high propagation time for cloud nodes
PRC_gv_beta  = PR_gv_beta * 4
BWC_gv_alpha = BW_gv_alpha  # Bandwidht is the same as sother nodes. 
BWC_gv_beta  = BW_gv_beta
IPT_g_mu     = 1000  # Instructions per Time gauss settings
IPT_g_sigma  = 200   # Instructions per Time gauss settings
MEM_ev_lambd = 1E-4  # Memory exponential distribution setting
SIM_DURATION = 600   # Time to run simulation
SIM_ITERS    = 1     # Number of iterations
P_SIBLING    = .5    # Probability for creating a link between siblings in a tree
NCOLONIES    = 10    # Default number of fog colonies

class FogRandomStaticPlacement(Placement):
    '''
    The class places services at random nodes that have minimum required MEM configuration. 
    '''

    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        pserver = PSERVER
        pots = []
        minmem = MINMEM    # Min 1 GB MEM

        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            if mem > minmem and random.random() <= pserver:
                pots.append(n)

        print(f'Server nodes for {app_name} ({len(pots)}): {pots}')
        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], pots)


class FogTreeSiblingConnectionPlacement(Placement):
    '''
    The class generate a tree structure from the network, with connections between siblings. 
    The service is placed in all nodes that have minimum requirements, and that is not a leaf node
    '''
    
    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        minmem = MINMEM
        pserver = PSERVER
        serverlist = []

        # Select nodes to act as servers
        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            if mem > minmem and sim.topology.G.degree(n) > 1 and random.random() <= pserver:
                serverlist.append(n)

        print(f'Server nodes ({len(serverlist)}): {serverlist}')
        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], serverlist)

class FogColonyPlacement(Placement):
    '''
    The class places servers in some of the nodes in some clusters, simulating busy nodes whithin a cluster.
    In clusters with no modules, it simulates a cluster where no nodes can process the request.
    It will not place a service in the orchestration node.
    '''
    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        minmem = MINMEM
        pserver = PSERVER
        serverlist = []

        for n in sim.topology.G.nodes():
            mem = sim.topology.G.nodes()[n]['MEM']
            controller = sim.topology.G.nodes()[n]['CONTROLLER']
            if n != controller and mem > minmem and random.random() <= pserver:
                serverlist.append(n)

        print(f'Server nodes ({len(serverlist)}): {serverlist}')
        for module in services.keys():
            idDES = sim.deploy_module(app_name, module, services[module], serverlist)


def deploy_random_lowresource_sources(sim):
    '''
    Deploy a number of sources, at the most memory constrained nodes, each with a message sending distribution. 
    '''
    nsources = int(nx.number_of_nodes(sim.topology.G) * NSOURCES)
    print('Number of sources: {nsources}')
    # make a list of sources, select the nsources with lowest MEM
    psources = [x for x in sorted(sim.topology.G.nodes(), key=lambda x: sim.topology.G.nodes()[x]['MEM'])][:nsources]

    for a in sim.apps:
        for i in psources:
            # Select a random message from app
            msg = sim.apps[a].get_message(random.sample(sorted(sim.apps[a].messages), 1)[0])
            dist = deterministic_distribution(100, name='Deterministic')
            idDES = sim.deploy_source(a, id_node=i, msg=msg, distribution=dist)

def deploy_colony_sources(sim):
    '''
    Deploy source in the service orchestration nodes, simulating a gateway or node contacting this node directly.
    '''
    nsources = int(nx.number_of_nodes(sim.topology.G) * NSOURCES)
    psources = [x for x in sorted(sim.topology.G.nodes(), key=lambda x: sim.topology.G.nodes()[x]['MEM']) if sim.topology.G.nodes()[x]['CONTROLLER'] == x][:nsources]

    for a in sim.apps:
        for n in psources:
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


def add_sibling_edge(graph, parent, skip, p, pr, bw, d):
    ''' 
    Add edges between siblings in a tree. This will generate new edges between siblings with new randomly selected attributes.
    '''
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
            if p >= random.random():
                pairs.append(pair)
        graph.add_edges_from(pairs)
        attrs = {pair: {'PR': random.gammavariate(*pr), 'BW': random.gammavariate(*bw)} for pair in pairs}
        nx.set_edge_attributes(graph, attrs)

    for child in children:
        add_sibling_edge(graph, child, skip, p, pr, bw, d+1)

def connect_children(tree, graph, grandp, parent, p):
    '''
    From the parent node, parent, in the tree, copy connection for children from graph with probability p.
    As the graph is undirected, exclude the grandparent node.
    '''
    n = list(tree.neighbors(parent))
    
    if grandp != None:
        n.remove(grandp)
    print(n, grandp, parent)
    if len(n) <= 1:
        return []

    edges = []
    for e in itertools.combinations(n, 2):
        if graph.has_edge(*e) and random.random() < p:
            edges.append(e)
    # print(edges)
    # tree.add_edges_from(edges)
    # attr = {edge: {'PR': graph.edges[edge[0], edge[1]]['PR'], 'BW': graph.edges[edge[0], edge[1]]['BW']} for edge in edges}
    # attr = {edge: {'PR': 1, 'BW': 1} for edge in edges}
    # nx.set_edge_attributes(tree, attr)

    for child in n:
        edges.extend(connect_children(tree, graph, parent, child, p))

    return edges

def subgraph_tree(topology, p, pr=None, bw=None):
    original_G = topology.G.copy()
    stg = nx.minimum_spanning_tree(topology.G, weight='PR')
    # for n in stg.nodes():
    #     print(f'Node: {n}: {stg.nodes()[n]}')
    # for e in stg.edges():
    #     print(f'Edge: {e}: {stg.edges()[e]}')
    
    # find center, and add connection between children, using the supplied parameters for the gamma distribution
    center = nx.center(stg)[0]

    # TODO: This should copy existing edges, not create new ones
    # add_sibling_edge(stg, center, set([center]), p, pr, bw, 0)
    edges = connect_children(stg, original_G, None, center, p)
    stg.add_edges_from(edges)
    attr = {edge: {'PR': original_G.edges[edge]['PR'], 'BW': original_G.edges[edge]['BW']} for edge in edges}
    nx.set_edge_attributes(stg, attr)
    
    topology.G = stg
    return original_G, stg


def subgraph_full(topology):
    return topology.G.copy(), topology.G


def subgraph_colony(topology, ncolonies, cloudpr, cloudbw, cloudattr):
    '''
    Based on the colony generating algorithm from Guerrero et al., 2018: On the influence of fog colonies positioning in fog application makespan.
    They concluded that the betweenness centrality was best for colony controller nodes.
    '''
    original_G = topology.G.copy()
    centr = nx.betweenness_centrality(topology.G)

    centr = {xx: centr[xx] for xx in sorted(centr, reverse=True, key=lambda x: centr[x])[:ncolonies]}
    print(f'Controllers: {centr}')
    # set an attribute on all nodes indicating if they are a controller node.
    iscontr = {n: False for n in original_G.nodes()}
    for n in centr:
        iscontr[n] = True

    # Segment nodes in accordance to the closest controller node
    closecontr = dict()
    for n in topology.G.nodes:
        if iscontr[n] == False:
            close = dict()
            for cn in centr:
                # Use a short message, a probe as distance measure
                close[cn] = nx.single_source_dijkstra(topology.G, n, cn, weight=lambda a, b, attr: 100 / (attr['BW'] * 1E6) + attr['PR']) 
            closecontr[n] = min(close, key=lambda x: close[x][0])
            print(f'Closest controller: {(n, closecontr[n])} {close[closecontr[n]]}')
            #pprint.pprint(close)
        else:
            closecontr[n] = n
    nx.set_node_attributes(topology.G, name='CONTROLLER', values=closecontr)

    # Remove all edges that crosses cluster
    for n in topology.G.nodes:
        if iscontr[n] == False:
            for nn in list(topology.G.neighbors(n)):
                if closecontr[n] != closecontr[nn]:
                    print(f'remove edge: {(n, nn)} {topology.G.edges()[(n,nn)]} {topology.G.nodes()[n]} {topology.G.nodes()[nn]}')
                    topology.G.remove_edge(n, nn)

    # Add a cloud server that are connected to all controllers, set a high propagation delay to make sure path is seldom through here
    cloud = int(max(topology.G.nodes) + 1)
    topology.G.add_node(cloud)
    attr = {cloud: cloudattr}
    print(attr)
    nx.set_node_attributes(topology.G, attr)
    topology.G.nodes[cloud]['CONTROLLER'] = cloud
    cedges = {(cloud, x): {'PR': random.gammavariate(*cloudpr), 'BW': random.gammavariate(*cloudbw)} for x in centr}
    topology.G.add_edges_from(cedges)
    nx.set_edge_attributes(topology.G, cedges)

    return original_G, topology.G


def find_correlation(graph, centrality, stats, filename):
    '''
    Find the best correlation between the centrality measures and the nodes containing data.
    '''
    # Make a dict of whether a node has been a server or not
    servs = {n for n in stats.df_vol['dst']}
    dserv = {x: False for x in graph.nodes()}
    for s in servs:
        dserv[s] = True
    print(f'Number of actual server nodes: {len(servs)} {servs}')
    
    # Plot the histogram of number of servers in centrality intervals

    fig = plt.figure()
    ncent = len(centrality)
    ax = []
    i = 1
    for c in centrality:
        k = min(centrality[c])
        print(f'minimum centrality measure for {c}: {k}')
        if k <= 0:
            k = .001 - k  # k is negative or zero
            centrality_scaled = [centrality[c][x]+k for x in centrality[c]]
        else:
            centrality_scaled = [centrality[c][x] for x in centrality[c]]
        centrality_scaled, l = spstats.boxcox(centrality_scaled)
        
        cent = [centrality_scaled[n] for n in dserv if dserv[n]]
        nodes = [centrality_scaled[n] for n in dserv if not dserv[n]]
        ax.append(fig.add_subplot(ncent, 3, i))
        ax.append(fig.add_subplot(ncent, 3, i+1))
        ax.append(fig.add_subplot(ncent, 3, i+2))
        ax[-3].hist(centrality_scaled, bins='auto', log=False)
        ax[-2].hist(nodes, bins='auto', log=False)
        ax[-3].set_ylabel(c)
        ax[-1].hist(cent, bins='auto', log=False)
        i += 3
        print(f'BoxCox lambda = {l}')
        # Any difference between the two populations?
    plt.tight_layout()
    plt.savefig(filename)

    fig = plt.figure()
    

def print_aggregated_results(results):
    '''
    Print the aggregated results from the dict, where each dict entry is the results from one simulation
    '''

def main(stop_time, graphgen, serviceplacement, sourcedeployment, subgraph, it, folder_results, folder_data):
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
    attrPR = {x: random.gammavariate(PR_gv_alpha, PR_gv_beta) for x in t.G.edges} # Gives mostly valuse between 0 and 1 second
    attrBW = {x: random.gammavariate(BW_gv_alpha, BW_gv_beta) for x in t.G.edges}  # BW given in Mb/s
    nx.set_edge_attributes(t.G, name='PR', values=attrPR)
    nx.set_edge_attributes(t.G, name='BW', values=attrBW)
    #print('PR and BW')
    fe.write('Edge;PR;BW\n')
    for e in t.G.edges():
        #print(f'e: {e}, PR: {t.G.edges()[e]["PR"]} BW: {t.G.edges()[e]["BW"]}')
        fe.write(f'{e};{t.G.edges()[e]["PR"]};{t.G.edges()[e]["BW"]}\n')
    fe.close()
 
    attrIPT = {x: abs(random.gauss(IPT_g_mu, IPT_g_sigma)) for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='IPT', values=attrIPT)
    
    # MEM (in MB) set at random from a n.e.d. 1E-4 gives very few at 100G, avg at 10G
    attrMEM = {x: int(random.expovariate(MEM_ev_lambd)) for x in t.G.nodes()}
    nx.set_node_attributes(t.G, name='MEM', values=attrMEM)
    #plt.hist([attrMEM[x] for x in attrMEM])
    #plt.show()
    #pprint.pprint(attrMEM)
    # print('IPT and MEM')
    fn.write('Node;IPT;MEM\n')
    for n in t.G.nodes():
        # print(f'n: {n}, IPT: {t.G.nodes()[n]["IPT"]}, MEM: {t.G.nodes()[n]["MEM"]}')
        fn.write(f'{n};{t.G.nodes()[n]["IPT"]};{t.G.nodes()[n]["MEM"]}\n')
    fn.close()

    nx.write_gexf(t.G, f'{folder_results}/{it:04}_original.gexf')
    nx.write_gpickle(t.G, f'{folder_results}/{it:04}_original.pkl')

    centrality = dict()
    centrality['degree']      = nx.degree_centrality(t.G)
    centrality['eigenvector'] = nx.eigenvector_centrality(t.G, max_iter=1000)
    centrality['closeness']   = nx.closeness_centrality(t.G)
    centrality['betweenness'] = nx.betweenness_centrality(t.G)
    centrality['harmonic']    = nx.harmonic_centrality(t.G)
    
    
    original_G, _ = subgraph['f'](t, **subgraph['args'])
    nx.write_gexf(t.G, f'{folder_results}/{it:04}_{subgraph["name"]}.gexf')
    nx.write_gpickle(t.G, f'{folder_results}/{it:04}_{subgraph["name"]}.pkl')

    with open(f'{folder_results}/{it:04}_{subgraph["name"]}.csv', 'w') as f:
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

    # Statistics
    stats = Stats(defaultPath=f'{folder_results}/{it:04}_sim_trace')
    stats.showVolatility(stop_time, t)

    # Correlation between server nodes and graph measures
    # Assume no knowledge of tree structure or clustering, use the original graph
    
    find_correlation(original_G, centrality, stats,  f'{folder_results}/{it:04}_correlation.pdf')


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
                       'args': {'n': DEFAULT_SIZE, 'p1': .05, 'p2': .01}},
                'gexf': {'f': nx.read_gexf,
                         'name': 'Existing GEXF graph',
                         'args': {'path': None, 'node_type': int}}}

    placement = {'randomstatic': FogRandomStaticPlacement,
                 'treesibling': FogTreeSiblingConnectionPlacement,
                 'colony': FogColonyPlacement}
    
    sourcedeployment = {'lowresource': deploy_random_lowresource_sources,
                        'leafnodes': deploy_leaf_nodes_sources,
                        'colony': deploy_colony_sources}

    subgraph = {'full': {'f': subgraph_full,
                         'name': 'full',
                         'args': dict()},
                'tree': {'f': subgraph_tree,
                         'name': 'tree',
                         'args': {'p': P_SIBLING, 'pr': (PR_gv_alpha, PR_gv_beta), 'bw': (BW_gv_alpha, BW_gv_beta)}},
                'colony': {'f': subgraph_colony,
                           'name': 'colony',
                           'args': {'ncolonies': NCOLONIES,
                                    'cloudpr': (PRC_gv_alpha, PRC_gv_beta),
                                    'cloudbw': (BWC_gv_alpha, BWC_gv_beta),
                                    'cloudattr': {'IPT': 1E9, 'MEM': 1E9}}}}

    aparse = argparse.ArgumentParser(description='Create a network, test various fog network structures')
    aparse.add_argument('-r', '--results', default='results', help='Directory to store results')
    aparse.add_argument('-d', '--datadir', default='data2', help='Directory for app and logging setup')
    aparse.add_argument('-g', '--graph', choices=graphgen.keys(), default='ba', help='Graph type to use for network creation')
    aparse.add_argument('-a', '--args', help='Named arguments for graph generator')
    aparse.add_argument('-p', '--placement', choices=placement.keys(), default='randomstatic', help='Fog service placement algorithm')
    aparse.add_argument('-s', '--source', choices=sourcedeployment.keys(), default='lowresource', help='Message source placement')
    aparse.add_argument('-u', '--subgraph', choices=subgraph.keys(), default='full', help='Subgraph for message routing in network')
    aparse.add_argument('-b', '--subargs', help='Subgraph arguments if others than defaults are given')
    args = aparse.parse_args()

    sim_duration = SIM_DURATION
    iterations = SIM_ITERS

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

    if args.subargs:
        a = json.loads(args.subargs)
        for e in a:
            subgraph[args.subgraph]['args'][e] = a[e]
            
    print(f'Parameters: Graph generator   = {graphgen[args.graph]["name"]}, args = {graphgen[args.graph]["args"]}')
    print(f'            Placement         = {args.placement}')
    print(f'            Source deployment = {args.source}')
    print(f'            Subgraph          = {subgraph[args.subgraph]["name"]}, args = {subgraph[args.subgraph]["args"]}')

    gcpy = graphgen[args.graph].copy()
    pcpy = placement.copy()
    scpy = sourcedeployment.copy()
    ucpy = subgraph[args.subgraph].copy()
    
    settings = {'graph': {args.graph: gcpy},
                'placement': {args.placement: pcpy[args.placement]},
                'sourcedeployment': {args.source: scpy[args.source]},
                'subgraph': {args.subgraph: ucpy},
                'datadir': args.datadir,
                'hardcoded': {'NSOURCES': NSOURCES,
                              'PSERVER': PSERVER,
                              'MINMEM': MINMEM,
                              'PR_gv_alpha': PR_gv_alpha,
                              'PR_gv_beta': PR_gv_beta,
                              'BW_gv_alpha': BW_gv_alpha,
                              'BW_gv_beta': BW_gv_beta,
                              'PRC_gv_alpha': PR_gv_alpha,
                              'PRC_gv_beta': PR_gv_beta,
                              'BWC_gv_alpha': BW_gv_alpha,
                              'BWC_gv_beta': BW_gv_beta,
                              'IPT_g_mu': IPT_g_mu,
                              'IPT_g_sigma': IPT_g_sigma,
                              'MEM_ev_lambd': MEM_ev_lambd,
                              'SIM_DURATION': SIM_DURATION,
                              'SIM_ITERS': SIM_ITERS,
                              'P_SIBLING': P_SIBLING,
                              'NCOLONIES': NCOLONIES}}
    # change pointers to strings
    settings['graph'][args.graph]['f'] = str(settings['graph'][args.graph]['f'])
    settings['placement'][args.placement] = str(settings['placement'][args.placement])
    settings['sourcedeployment'][args.source] = str(settings['sourcedeployment'][args.source])
    settings['subgraph'][args.subgraph]['f'] = str(settings['subgraph'][args.subgraph]['f'])
    
    with open(f'{args.results}/settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

    result = dict()
    for i in range(iterations):
        tstart = time.time()
        result[i] = main(sim_duration,
                         graphgen[args.graph],
                         placement[args.placement],
                         sourcedeployment[args.source],
                         subgraph[args.subgraph],
                         i,
                         folder_results,
                         folder_data)
        print(f'\n--- Iteration: {i}: {time.time() - tstart} seconds ---')

    print_aggregated_results(result)
    
    print('simulation finished')
