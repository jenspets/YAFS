"""
This class is for setting the volatility of each node in the network.
"""

import logging
import random
import networkx as nx

class Volatility(object):
    # just choose some random starting point for these constants
    SOURCE = 'SRC'
    PROXY = 'PRX'
    SINK = 'SNK'
    SERVER = 'SRV'
    
    def __init__(self, app, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.app = app


    def set_unlinkdistr(self, distr):
        pass


    def get_unlinktime(self, message, vtype, node=None):
        """ returrn a sample time between creation and unlink/ deletion, geven a tyep of node"""
        return 1.0


    def set_erasedistr(self, distr):
        pass


    def get_erasetime(self, message, vtype, node=None):
        """ return a sample  time between the unlinking of data and the erasure of the data. """
        return 1.0


class ExponentialVolatility(Volatility):
    """ A class where the volatility with an Negative exponential distribution. Parameter to the distribution is lambds. """
    def __init__(self, app, topology, logger=None):
        self.dname = 'default_vol'  # Name for default dictionary item if no message name is given
        self.etime_type = {self.dname: {}}
        self.etime_node = {self.dname: {}}
        self.etime = {self.dname: 0}
        self.utime_type = {self.dname: {}}
        self.utime_node = {self.dname: {}}
        self.utime = {self.dname: 0}
        self.t = topology
        super().__init__(app, logger)

    def set_unlinkdistr(self, lmbd, message_name=None, vtype=None, node=None):
        if node:
            # Set the volatility for a specific node
            if message_name:
                if message_name not in self.utime_node:
                    self.utime_node[message_name] = {}
                self.utime_node[message_name][node] = lmbd
            else:
                self.utime_node[self.dname][node] = lmbd
        elif vtype:
            if message_name:
                if message_name not in self.utime_type:
                    self.utime_type[message_name] = {}
                self.utime_type[message_name][vtype] = lmbd
            else:
                self.utime_type[self.dname][vtype] = lmbd
        else:
            if message_name:
                self.utime[message_name] = lmbd
            else:
                self.utime[self.dname] = lmbd

    def set_erasedistr(self, lmbd, message_name=None, vtype=None, node=None):
        if node:
            # Set the volatility for a specific node
            if message_name:
                if message_name not in self.etime_node:
                    self.etime_node[message_name] = {}
                self.etime_node[message_name][node] = lmbd
            else:
                self.etime_node[self.dname][node] = lmbd
        elif vtype:
            if message_name:
                if message_name not in self.etime_type:
                    self.etime_type[message_name] = {}
                self.etime_type[message_name][vtype] = lmbd
            else:
                self.etime_type[self.dname][vtype] = lmbd
        else:
            if message_name:
                self.etime[message_name] = lmbd
            else:
                self.etime[self.dname] = lmbd

    def get_unlinktime(self, message, node=None, vtype=None):
        m = message.name
        lmbd = 0

        if node and m in self.utime_node and node in self.utime_node[m]:
            lmbd = self.utime[m][node]
        elif vtype and m in self.utime_type and vtype in self.utime_type[m][vtype]:
            lmbd = self.utime_type[m][vtype]
        elif node and node in self.utime_node[self.dname]:
            lmbd = self.utime_node[self.dname][node]
        elif vtype and vtype in self.utime_type[self.dname]:
            lmbd = self.utime_type[self.dname][vtype]
        elif m in self.utime:
            lmbd = self.utime[m]
        else:
            lmbd = self.utime[self.dname]

        return random.expovariate(lmbd)

    def get_erasetime(self, message, node, vtype=None):
        
        m = message.name
        lmbd = 0

        # Modify lambda based or MEM 
        # Scaling function selected to scale retention time down for resource constrained nodes
        # scaling factor selected to be .1 to 1
        hi = 1
        lo = .1
        
        mmax = max(nx.get_node_attributes(self.t.G, 'MEM').values())
        scale = lo + (hi - lo) * (nx.get_node_attributes(self.t.G, 'MEM')[node] / mmax)

        if node and m in self.etime_node and node in self.etime_node[m]:
            lmbd = self.etime_node[m][node] / scale
        elif vtype and m in self.etime_type and vtype in self.etime_type[m]:
            lmbd = self.etime_type[m][vtype] / scale
        elif node and node in self.etime_node[self.dname]:
            lmbd = self.etime_node[self.dname][node] / scale
        elif vtype and vtype in self.etime_type[self.dname]:
            lmbd = self.etime_type[self.dname][vtype] / scale
        elif m in self.etime:
            lmbd = self.etime[m] / scale
        else:
            lmbd = self.etime[self.dname] / scale
        # print(f'Node: {node}, scale: {scale}, lambda: {lmbd}, MEM: {nx.get_node_attributes(self.t.G, "MEM")[node]}')
        return random.expovariate(lmbd)


class UniformVolatility(Volatility):
    """ Volatility drawn from a uniform distribution, set by each type of node."""
    
    def __init__(self, app, logger=None):
        self.dname = 'default_vol'  # Name for default dictionary item if no message name is given
        self.etime_node = {self.dname: {}}
        self.etime_type = {self.dname: {}}
        self.etime = {self.dname: (0, 0)}
        self.utime_node = {self.dname: {}}
        self.utime_type = {self.dname: {}}
        self.utime = {self.dname: (0, 0)}
        super().__init__(app, logger)

    def set_erasedistr(self, time_min, time_max, message_name=None, vtype=None, node=None):
        """ Set the distribution for time between unlink and erasure."""
        tmin = min((time_min, time_max))
        tmax = max((time_min, time_max))

        if node:
            # Set the volatility for a specific node
            if message_name:
                if message_name not in self.etime_node:
                    self.etime_node[message_name] = {}
                self.etime_node[message_name][node] = (tmin, tmax)
            else:
                self.etime_node[self.dname][node] = (tmin, tmax)
        elif vtype:
            if message_name:
                if message_name not in self.etime_type:
                    self.etime_type[message_name] = {}
                self.etime_type[message_name][vtype] = (tmin, tmax)
            else:
                self.etime_type[self.dname][vtype] = (tmin, tmax)
        else:
            if message_name:
                self.etime[message_name] = (tmin, tmax)
            else:
                self.etime[self.dname] = (tmin, tmax)

    def set_unlinkdistr(self, time_min, time_max, message_name=None, vtype=None, node=None):
        """ Set the distribution for time between creation and unlink of data. """
        tmin = min((time_min, time_max))
        tmax = max((time_min, time_max))

        if node:
            if message_name:
                if message_name not in self.utime_node:
                    self.utime_node[message_name] = {}
                self.utime_node[message_name][node] = (tmin, tmax)
            else:
                self.utime_node[self.dname][node] = (tmin, tmax)
        elif vtype:
            if message_name:
                if message_name not in self.etime_type:
                    self.etime_type[message_name] = {}
                self.etime_type[message_name][vtype] = (tmin, tmax)
            else:
                self.utime_type[self.dname][vtype] = (tmin, tmax)
        else:
            if message_name:
                self.utime[message_name] = (tmin, tmax)
            else:
                self.utime[self.dname] = (tmin, tmax)

    def get_erasetime(self, message, vtype=None, node=None):
        m = message.name
        limits = (0, 0)

        if node and m in self.etime_node and node in self.etime_node[m]:
            limits = self.etime_node[m][node]
        elif vtype and m in self.etime_type and vtype in self.etime_type[m]:
            limits = self.etime_type[m][vtype]
        elif node and node in self.etime_node[self.dname]:
            limits = self.etime_node[self.dname][node]
        elif vtype and vtype in self.etime_type[self.dname]:
            limits = self.etime_type[self.dname][vtype]
        elif m in self.etime:
            limits = self.etime[m]
        else:
            limits = self.etime[self.dname]

        return random.uniform(limits[0], limits[1])


    def get_unlinktime(self, message, vtype=None, node=None):
        m = message.name
        limits = (0, 0)

        if node and m in self.utime_node and node in self.utime_node[m]:
            limits = self.utime[m][node]
        elif vtype and m in self.utime_type and vtype in self.utime_type[m][vtype]:
            limits = self.utime_type[m][vtype]
        elif node and node in self.utime_node[self.dname]:
            limits = self.utime_node[self.dname][node]
        elif vtype and vtype in self.utime_type[self.dname]:
            limits = self.utime_type[self.dname][vtype]
        elif m in self.utime:
            limits = self.utime[m]
        else:
            limits = self.utime[self.dname]

        return random.uniform(limits[0], limits[1])

class FixedVolatility(Volatility):
    """ Just a single, set volatility for all nodes. For testing. """

    def get_unlinktime(self, message, vtype=None, node=None):
        return self.ultime

    def get_erasetime(self, message, vtype=None, node=None):
        return self.etime

    def set_unlinkdistr(self, time):
        self.ultime = time

    def set_erasedistr(self, time):
        self.etime = time
