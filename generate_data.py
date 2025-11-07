#!/usr/bin/env python3

"""
Computer Networks Lab Project
Simulated TCP Performance Data Generator
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

def generate_tcp_tahoe_data(network_type, duration=30, interval=1):
    """
    Generate simulated TCP Tahoe performance data
    
    TCP Tahoe characteristics:
    - Slow start
    - Congestion avoidance
    - Fast retransmit
    - Resets to slow start on packet loss
    """
    # Time points
    time_points = np.arange(0, duration, interval)
    
    # Base throughput parameters
    if network_type == 'lan':
        base_throughput = 90  # Mbps for LAN
        noise_level = 2       # Lower noise in LAN
        drop_probability = 0.1  # Lower drop probability in LAN
    else:  # wlan
        base_throughput = 40  # Mbps for WLAN
        noise_level = 5       # Higher noise in WLAN
        drop_probability = 0.3  # Higher drop probability in WLAN
    
    # Initialize throughput array
    throughput = np.zeros_like(time_points, dtype=float)
    
    # Simulate TCP Tahoe behavior
    current_throughput = 10  # Starting throughput
    
    for i, t in enumerate(time_points):
        # Add some random noise
        noise = np.random.normal(0, noise_level)
        
        # Simulate packet loss events
        if np.random.random() < drop_probability and i > 0:
            # TCP Tahoe drops to slow start on packet loss
            current_throughput = 10
        else:
            if current_throughput < base_throughput * 0.7:
                # Slow start phase - exponential growth
                current_throughput = min(current_throughput * 1.5, base_throughput)
            else:
                # Congestion avoidance phase - linear growth
                current_throughput = min(current_throughput + 2, base_throughput)
        
        # Set throughput with noise
        throughput[i] = max(0, current_throughput + noise)
    
    return time_points, throughput

def generate_tcp_reno_data(network_type, duration=30, interval=1):
    """
    Generate simulated TCP Reno performance data
    
    TCP Reno characteristics:
    - Slow start
    - Congestion avoidance
    - Fast retransmit
    - Fast recovery (main difference from Tahoe)
    """
    # Time points
    time_points = np.arange(0, duration, interval)
    
    # Base throughput parameters
    if network_type == 'lan':
        base_throughput = 92  # Mbps for LAN
        noise_level = 2       # Lower noise in LAN
        drop_probability = 0.1  # Lower drop probability in LAN
    else:  # wlan
        base_throughput = 42  # Mbps for WLAN
        noise_level = 5       # Higher noise in WLAN
        drop_probability = 0.3  # Higher drop probability in WLAN
    
    # Initialize throughput array
    throughput = np.zeros_like(time_points, dtype=float)
    
    # Simulate TCP Reno behavior
    current_throughput = 10  # Starting throughput
    
    for i, t in enumerate(time_points):
        # Add some random noise
        noise = np.random.normal(0, noise_level)
        
        # Simulate packet loss events
        if np.random.random() < drop_probability and i > 0:
            # TCP Reno cuts window in half on packet loss (fast recovery)
            current_throughput = max(current_throughput / 2, 10)
        else:
            if current_throughput < base_throughput * 0.7:
                # Slow start phase - exponential growth
                current_throughput = min(current_throughput * 1.5, base_throughput)
            else:
                # Congestion avoidance phase - linear growth
                current_throughput = min(current_throughput + 2, base_throughput)
        
        # Set throughput with noise
        throughput[i] = max(0, current_throughput + noise)
    
    return time_points, throughput

def generate_tcp_cubic_data(network_type, duration=30, interval=1):
    """
    Generate simulated TCP Cubic performance data
    
    TCP Cubic characteristics:
    - Window growth is a cubic function of time
    - Less aggressive at the beginning, more aggressive later
    - Better utilization of high bandwidth networks
    """
    # Time points
    time_points = np.arange(0, duration, interval)
    
    # Base throughput parameters
    if network_type == 'lan':
        base_throughput = 95  # Mbps for LAN (higher than Reno)
        noise_level = 2       # Lower noise in LAN
        drop_probability = 0.1  # Lower drop probability in LAN
    else:  # wlan
        base_throughput = 45  # Mbps for WLAN (higher than Reno)
        noise_level = 5       # Higher noise in WLAN
        drop_probability = 0.3  # Higher drop probability in WLAN
    
    # Initialize throughput array
    throughput = np.zeros_like(time_points, dtype=float)
    
    # Simulate TCP Cubic behavior
    current_throughput = 10  # Starting throughput
    last_drop_time = -5      # Time since last drop (negative means before start)
    
    for i, t in enumerate(time_points):
        # Add some random noise
        noise = np.random.normal(0, noise_level)
        
        # Simulate packet loss events
        if np.random.random() < drop_probability and i > 0:
            # TCP Cubic reduces window on packet loss but not as drastically as Reno
            current_throughput = max(current_throughput * 0.7, 10)
            last_drop_time = t
        else:
            # Cubic growth function based on time since last drop
            time_since_drop = t - last_drop_time
            if time_since_drop > 0:
                # Cubic growth function: K(t-T)^3 + Wmax
                # Simplified version
                growth = 0.2 * (time_since_drop ** 3) + 5
                current_throughput = min(current_throughput + growth, base_throughput)
            else:
                # Initial growth
                current_throughput = min(current_throughput * 1.2, base_throughput)
        
        # Set throughput with noise
        throughput[i] = max(0, current_throughput + noise)
    
    return time_points, throughput

def generate_all_data():
    """
    Generate simulated data for all TCP variants and network types
    """
    # Create directories for results
    os.makedirs('results/lan', exist_ok=True)
    os.makedirs('results/wlan', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Network types
    network_types = ['lan', 'wlan']
    
    # Generate data for each network type
    for network in network_types:
        print(f"*** Generating simulated data for {network.upper()} network")
        
        # Generate TCP Tahoe data (using Reno as proxy)
        time_tahoe, throughput_tahoe = generate_tcp_tahoe_data(network)
        
        # Generate TCP Reno data
        time_reno, throughput_reno = generate_tcp_reno_data(network)
        
        # Generate TCP Cubic data
        time_cubic, throughput_cubic = generate_tcp_cubic_data(network)
        
        # Save data to CSV files
        tahoe_df = pd.DataFrame({'Time': time_tahoe, 'Throughput': throughput_tahoe})
        reno_df = pd.DataFrame({'Time': time_reno, 'Throughput': throughput_reno})
        cubic_df = pd.DataFrame({'Time': time_cubic, 'Throughput': throughput_cubic})
        
        tahoe_df.to_csv(f'results/{network}/tahoe_throughput.csv', index=False)
        reno_df.to_csv(f'results/{network}/reno_throughput.csv', index=False)
        cubic_df.to_csv(f'results/{network}/cubic_throughput.csv', index=False)
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_tahoe, throughput_tahoe, label='TCP Tahoe')
        plt.plot(time_reno, throughput_reno, label='TCP Reno')
        plt.plot(time_cubic, throughput_cubic, label='TCP Cubic')
        
        plt.title(f'TCP Variants Performance Comparison - {network.upper()}')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (Mbps)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'plots/tcp_comparison_{network}.png')
        plt.close()
        
        # Create individual plots
        plt.figure(figsize=(8, 5))
        plt.plot(time_tahoe, throughput_tahoe, 'b-')
        plt.title(f'TCP Tahoe Performance - {network.upper()}')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.savefig(f'plots/tcp_tahoe_{network}.png')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        plt.plot(time_reno, throughput_reno, 'g-')
        plt.title(f'TCP Reno Performance - {network.upper()}')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.savefig(f'plots/tcp_reno_{network}.png')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        plt.plot(time_cubic, throughput_cubic, 'r-')
        plt.title(f'TCP Cubic Performance - {network.upper()}')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.savefig(f'plots/tcp_cubic_{network}.png')
        plt.close()
        
        # Generate simulated congestion window data
        print(f"*** Generating simulated congestion window data for {network.upper()} network")
        
        # Time points with higher resolution for cwnd
        cwnd_time = np.arange(0, 30, 0.1)
        
        # Tahoe cwnd behavior
        cwnd_tahoe = np.zeros_like(cwnd_time)
        current_cwnd = 1
        ssthresh = 64
        
        for i, t in enumerate(cwnd_time):
            if np.random.random() < 0.02 and i > 0:  # Packet loss
                ssthresh = max(current_cwnd / 2, 2)
                current_cwnd = 1  # Tahoe resets to 1
            else:
                if current_cwnd < ssthresh:  # Slow start
                    current_cwnd = min(current_cwnd * 2, 100)
                else:  # Congestion avoidance
                    current_cwnd = min(current_cwnd + 1/current_cwnd, 100)
            
            cwnd_tahoe[i] = current_cwnd
        
        # Reno cwnd behavior
        cwnd_reno = np.zeros_like(cwnd_time)
        current_cwnd = 1
        ssthresh = 64
        
        for i, t in enumerate(cwnd_time):
            if np.random.random() < 0.02 and i > 0:  # Packet loss
                ssthresh = max(current_cwnd / 2, 2)
                current_cwnd = ssthresh  # Reno goes to ssthresh (fast recovery)
            else:
                if current_cwnd < ssthresh:  # Slow start
                    current_cwnd = min(current_cwnd * 2, 100)
                else:  # Congestion avoidance
                    current_cwnd = min(current_cwnd + 1/current_cwnd, 100)
            
            cwnd_reno[i] = current_cwnd
        
        # Cubic cwnd behavior
        cwnd_cubic = np.zeros_like(cwnd_time)
        current_cwnd = 1
        wmax = 0
        
        for i, t in enumerate(cwnd_time):
            if np.random.random() < 0.02 and i > 0:  # Packet loss
                wmax = current_cwnd
                current_cwnd = max(current_cwnd * 0.7, 1)  # Cubic reduces less drastically
            else:
                if wmax == 0:  # Initial phase
                    current_cwnd = min(current_cwnd * 1.5, 100)
                else:  # Cubic growth
                    # Simplified cubic function
                    k = 0.4
                    current_cwnd = min(k * ((t - cwnd_time[i-1]) ** 3) + wmax, 100)
            
            cwnd_cubic[i] = current_cwnd
        
        # Save cwnd plots
        plt.figure(figsize=(10, 6))
        plt.plot(cwnd_time, cwnd_tahoe, label='TCP Tahoe')
        plt.plot(cwnd_time, cwnd_reno, label='TCP Reno')
        plt.plot(cwnd_time, cwnd_cubic, label='TCP Cubic')
        
        plt.title(f'TCP Congestion Window Comparison - {network.upper()}')
        plt.xlabel('Time (s)')
        plt.ylabel('Congestion Window Size (packets)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'plots/tcp_cwnd_comparison_{network}.png')
        plt.close()
        
        # Create simulated iperf logs
        print(f"*** Creating simulated iperf logs for {network.upper()} network")
        
        # Tahoe iperf log
        with open(f'results/{network}/iperf_client_{network}_tahoe.log', 'w') as f:
            f.write(f"------------------------------------------------------------\n")
            f.write(f"Client connecting to 10.0.0.2, TCP port 5001\n")
            f.write(f"TCP window size: 85.3 KByte (default)\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(f"[ ID] Interval       Transfer     Bandwidth\n")
            
            for i in range(len(time_tahoe)):
                if i > 0:
                    f.write(f"[  3] {i-1:.1f}-{i:.1f} sec    {throughput_tahoe[i]*0.125:.1f} MBytes  {throughput_tahoe[i]:.1f} Mbits/sec\n")
            
            f.write(f"[  3] 0.0-30.0 sec    {np.mean(throughput_tahoe)*3.75:.1f} MBytes  {np.mean(throughput_tahoe):.1f} Mbits/sec\n")
        
        # Reno iperf log
        with open(f'results/{network}/iperf_client_{network}_reno.log', 'w') as f:
            f.write(f"------------------------------------------------------------\n")
            f.write(f"Client connecting to 10.0.0.2, TCP port 5001\n")
            f.write(f"TCP window size: 85.3 KByte (default)\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(f"[ ID] Interval       Transfer     Bandwidth\n")
            
            for i in range(len(time_reno)):
                if i > 0:
                    f.write(f"[  3] {i-1:.1f}-{i:.1f} sec    {throughput_reno[i]*0.125:.1f} MBytes  {throughput_reno[i]:.1f} Mbits/sec\n")
            
            f.write(f"[  3] 0.0-30.0 sec    {np.mean(throughput_reno)*3.75:.1f} MBytes  {np.mean(throughput_reno):.1f} Mbits/sec\n")
        
        # Cubic iperf log
        with open(f'results/{network}/iperf_client_{network}_cubic.log', 'w') as f:
            f.write(f"------------------------------------------------------------\n")
            f.write(f"Client connecting to 10.0.0.2, TCP port 5001\n")
            f.write(f"TCP window size: 85.3 KByte (default)\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(f"[ ID] Interval       Transfer     Bandwidth\n")
            
            for i in range(len(time_cubic)):
                if i > 0:
                    f.write(f"[  3] {i-1:.1f}-{i:.1f} sec    {throughput_cubic[i]*0.125:.1f} MBytes  {throughput_cubic[i]:.1f} Mbits/sec\n")
            
            f.write(f"[  3] 0.0-30.0 sec    {np.mean(throughput_cubic)*3.75:.1f} MBytes  {np.mean(throughput_cubic):.1f} Mbits/sec\n")
    
    print("*** All simulated data generated successfully")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate all simulated data
    generate_all_data()
