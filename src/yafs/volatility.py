"""
This class is for setting the volatility of each node in the network.
"""

import logging
import random

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


class UniformVolatility(Volatility):
    """ Volatility drawn from a uniform distribution, set by each type of node. """

    def __init__(self, app, logger=None):
        self.etime_node = {}
        self.etime_type = {}
        self.etime = (0, 0)
        self.utime_node = {}
        self.utime_type = {}
        self.utime = (0, 0)
        super().__init__(app, logger)
        
    def set_erasedistr(self, time_min, time_max, vtype=None, node=None):
        tmin = min((time_min, time_max))
        tmax = max((time_min, time_max))

        if node:
            # Set the volatility for a specific node
            self.etime_node[node] = (tmin, tmax)
        elif vtype:
            self.etime_type[vtype] = (tmin, tmax)
        else:
            self.etime = (tmin, tmax)

    def set_unlinkdistr(self, time_min, time_max, vtype=None, node=None):
        tmin = min((time_min, time_max))
        tmax = max((time_min, time_max))

        if node:
            self.utime_node[node] = (tmin, tmax)
        elif vtype:
            self.utime_type[vtype] = (tmin, tmax)
        else:
            self.utime = (tmin, tmax)

    def get_erasetime(self, message, vtype=None, node=None):
        if node and node in self.etime_node:
            t = random.uniform(self.etime_node[node][0], self.etime_node[node][1])
        elif vtype and vtype in self.etime_type:
            t = random.uniform(self.etime_type[vtype][0], self.etime_type[vtype][1])
        else:
            t = random.uniform(self.etime[0], self.etime[1])

        return t

    def get_unlinktime(self, message, vtype=None, node=None):
        if node and node in self.utime_node:
            t = random.uniform(self.utime_node[node][0], self.utime_node[node][1])
        elif vtype and vtype in self.utime_type:
            t = random.uniform(self.utime_type[vtype][0], self.utime_type[vtype][1])
        else:
            t = random.uniform(self.utime[0], self.utime[1])

        return t


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
