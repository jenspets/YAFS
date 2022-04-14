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
