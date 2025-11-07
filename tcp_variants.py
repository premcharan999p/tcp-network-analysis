#!/usr/bin/env python3

"""
Computer Networks Lab Project
TCP Variants Configuration and Testing Script
"""

import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

def check_tcp_variants():
    """
    Check available TCP congestion control algorithms
    """
    print("*** Checking available TCP congestion control algorithms")
    result = subprocess.run(['sysctl', 'net.ipv4.tcp_available_congestion_control'], 
                           capture_output=True, text=True)
    print(result.stdout)
    
    print("*** Checking current TCP congestion control algorithm")
    result = subprocess.run(['sysctl', 'net.ipv4.tcp_congestion_control'], 
                           capture_output=True, text=True)
    print(result.stdout)
    
    return result.stdout.strip().split('=')[1].strip()

def configure_tcp_variant(variant):
    """
    Configure TCP congestion control algorithm
    """
    print(f"*** Configuring TCP congestion control algorithm to {variant}")
    result = subprocess.run(['sudo', 'sysctl', '-w', f'net.ipv4.tcp_congestion_control={variant}'], 
                           capture_output=True, text=True)
    print(result.stdout)
    
    # Verify the change
    current = check_tcp_variants()
    if variant in current:
        print(f"*** Successfully configured TCP variant to {variant}")
        return True
    else:
        print(f"*** Failed to configure TCP variant to {variant}")
        return False

def configure_tcp_parameters():
    """
    Configure additional TCP parameters for testing
    """
    # Enable TCP window scaling
    os.system('sudo sysctl -w net.ipv4.tcp_window_scaling=1')
    
    # Set initial congestion window size
    os.system('sudo sysctl -w net.ipv4.tcp_initial_cwnd=10')
    
    # Enable TCP timestamps
    os.system('sudo sysctl -w net.ipv4.tcp_timestamps=1')
    
    # Enable TCP SACK (Selective Acknowledgment)
    os.system('sudo sysctl -w net.ipv4.tcp_sack=1')
    
    # Set TCP retransmission parameters
    os.system('sudo sysctl -w net.ipv4.tcp_retries2=8')
    
    print("*** TCP parameters configured for testing")

def plot_tcp_variants_comparison():
    """
    Create a placeholder function for plotting TCP variants comparison
    This will be populated with actual data after running the tests
    """
    print("*** Creating placeholder for TCP variants comparison plot")
    
    # Create a directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Create a placeholder plot
    plt.figure(figsize=(10, 6))
    plt.title('TCP Variants Performance Comparison (Placeholder)')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    
    # Placeholder data
    x = np.arange(0, 30)
    y_tahoe = np.zeros(30)  # Will be replaced with actual data
    y_reno = np.zeros(30)   # Will be replaced with actual data
    y_cubic = np.zeros(30)  # Will be replaced with actual data
    
    plt.plot(x, y_tahoe, label='TCP Tahoe')
    plt.plot(x, y_reno, label='TCP Reno')
    plt.plot(x, y_cubic, label='TCP Cubic')
    
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/tcp_variants_comparison_placeholder.png')
    plt.close()

if __name__ == '__main__':
    # Check available TCP variants
    current_variant = check_tcp_variants()
    print(f"Current TCP variant: {current_variant}")
    
    # Configure TCP parameters
    configure_tcp_parameters()
    
    # Test TCP Reno (as a substitute for Tahoe)
    configure_tcp_variant('reno')
    
    # Test TCP Reno
    configure_tcp_variant('reno')
    
    # Test TCP Cubic
    configure_tcp_variant('cubic')
    
    # Create placeholder plot
    plot_tcp_variants_comparison()
    
    print("*** TCP variants configuration completed")
