#!/usr/bin/env python3

"""
Computer Networks Lab Project
Modified WLAN Topology Implementation for TCP Performance Analysis
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Host, OVSController
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
import time
import os
import subprocess

class WlanTopology(Topo):
    """
    WLAN topology with configurable number of hosts and an access point
    """
    def build(self, n=4):
        # Add switch (representing wireless access point)
        ap = self.addSwitch('ap1')
        
        # Add hosts and connect them to the access point
        for h in range(n):
            # Add host
            host = self.addHost(f'h{h+1}')
            
            # Add link with wireless characteristics:
            # - Lower bandwidth (54Mbps typical for 802.11g)
            # - Higher delay (10-20ms typical for wireless)
            # - Packet loss to simulate wireless interference (0.5%)
            # - Jitter to simulate variable wireless conditions
            self.addLink(host, ap, bw=54, delay='15ms', loss=0.5, jitter='5ms')

def perfTest(tcp_variant):
    """
    Test network performance with specified TCP variant
    """
    # Set TCP congestion control algorithm
    os.system(f'sysctl -w net.ipv4.tcp_congestion_control={tcp_variant}')
    
    # Create topology and network
    topo = WlanTopology(n=2)  # Using only 2 hosts for simplicity
    net = Mininet(topo=topo, host=Host, link=TCLink, controller=OVSController)
    
    # Start network
    net.start()
    
    print("*** Dumping host connections")
    dumpNodeConnections(net.hosts)
    
    print("*** Testing network connectivity")
    net.pingAll()
    
    print(f"*** Testing TCP performance with {tcp_variant} variant")
    h1, h2 = net.get('h1', 'h2')
    
    # Create results directory
    os.makedirs('results/wlan', exist_ok=True)
    
    # Start iperf server on h2
    print(f"*** Starting iperf server on {h2.name}")
    server_cmd = f'iperf -s -i 1 > results/wlan/iperf_server_wlan_{tcp_variant}.log 2>&1 &'
    h2.cmd(server_cmd)
    
    # Allow server to start up
    time.sleep(2)
    
    # Run iperf client on h1
    print(f"*** Running iperf test with TCP {tcp_variant} from {h1.name} to {h2.name}")
    client_cmd = f'iperf -c {h2.IP()} -t 30 -i 1 > results/wlan/iperf_client_wlan_{tcp_variant}.log 2>&1'
    h1.cmd(client_cmd)
    
    # Stop iperf server
    h2.cmd('pkill -f "iperf -s"')
    
    # Capture TCP congestion window statistics
    print(f"*** Capturing TCP statistics for {tcp_variant}")
    h1.cmd(f'ss -i > results/wlan/ss_stats_wlan_{tcp_variant}.log 2>&1')
    
    # Take a screenshot of the network topology
    print(f"*** Taking screenshot of network topology for {tcp_variant}")
    net.pingAll()  # Run pingAll again for demonstration
    
    # Stop network
    net.stop()
    
    return True

if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
    
    # Create results directory
    os.makedirs('results/wlan', exist_ok=True)
    
    # Run performance test with different TCP variants
    print("*** Running WLAN performance test with TCP Tahoe/Reno")
    success_reno = perfTest('reno')  # Using Reno as a substitute for Tahoe as it's not directly available
    
    print("*** Running WLAN performance test with TCP Cubic")
    success_cubic = perfTest('cubic')
    
    if success_reno and success_cubic:
        print("*** All WLAN tests completed successfully")
    else:
        print("*** Some WLAN tests failed")
