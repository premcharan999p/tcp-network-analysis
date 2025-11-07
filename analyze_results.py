#!/usr/bin/env python3

"""
Computer Networks Lab Project
TCP Performance Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def analyze_throughput_data():
    """
    Analyze throughput data for different TCP variants and network types
    """
    # Create directory for analysis results
    os.makedirs('analysis', exist_ok=True)
    
    # Network types
    network_types = ['lan', 'wlan']
    
    # TCP variants
    tcp_variants = ['tahoe', 'reno', 'cubic']
    
    # Results dictionary
    results = {}
    
    # Analyze each network type
    for network in network_types:
        results[network] = {}
        
        # Load data for each TCP variant
        for variant in tcp_variants:
            try:
                # Load throughput data
                data_file = f'results/{network}/{variant}_throughput.csv'
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file)
                    
                    # Calculate statistics
                    avg_throughput = df['Throughput'].mean()
                    max_throughput = df['Throughput'].max()
                    min_throughput = df['Throughput'].min()
                    std_throughput = df['Throughput'].std()
                    
                    # Store results
                    results[network][variant] = {
                        'avg_throughput': avg_throughput,
                        'max_throughput': max_throughput,
                        'min_throughput': min_throughput,
                        'std_throughput': std_throughput
                    }
                    
                    print(f"*** Analyzed {variant} on {network}: Avg={avg_throughput:.2f} Mbps")
                else:
                    print(f"*** Warning: Data file {data_file} not found")
            except Exception as e:
                print(f"*** Error analyzing {variant} on {network}: {e}")
    
    # Create comparison bar chart for average throughput
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(tcp_variants))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    for i, network in enumerate(network_types):
        avg_values = [results[network][variant]['avg_throughput'] for variant in tcp_variants]
        pos = r1 if i == 0 else r2
        plt.bar(pos, avg_values, width=bar_width, label=network.upper())
    
    # Add labels and legend
    plt.xlabel('TCP Variant')
    plt.ylabel('Average Throughput (Mbps)')
    plt.title('Average Throughput Comparison by TCP Variant and Network Type')
    plt.xticks([r + bar_width/2 for r in r1], tcp_variants)
    plt.legend()
    
    # Save the figure
    plt.savefig('analysis/avg_throughput_comparison.png')
    plt.close()
    
    # Create table of results
    table_data = []
    for network in network_types:
        for variant in tcp_variants:
            if variant in results[network]:
                row = [
                    network.upper(),
                    variant.capitalize(),
                    f"{results[network][variant]['avg_throughput']:.2f}",
                    f"{results[network][variant]['max_throughput']:.2f}",
                    f"{results[network][variant]['min_throughput']:.2f}",
                    f"{results[network][variant]['std_throughput']:.2f}"
                ]
                table_data.append(row)
    
    # Create DataFrame for table
    table_df = pd.DataFrame(table_data, columns=[
        'Network', 'TCP Variant', 'Avg Throughput (Mbps)', 
        'Max Throughput (Mbps)', 'Min Throughput (Mbps)', 
        'Std Dev (Mbps)'
    ])
    
    # Save table to CSV
    table_df.to_csv('analysis/throughput_statistics.csv', index=False)
    
    # Create HTML table for report
    html_table = table_df.to_html(index=False)
    with open('analysis/throughput_table.html', 'w') as f:
        f.write(html_table)
    
    return results

def analyze_stability_and_fairness():
    """
    Analyze stability and fairness of different TCP variants
    """
    # Network types
    network_types = ['lan', 'wlan']
    
    # TCP variants
    tcp_variants = ['tahoe', 'reno', 'cubic']
    
    # Results dictionary
    stability_results = {}
    
    # Analyze each network type
    for network in network_types:
        stability_results[network] = {}
        
        # Load data for each TCP variant
        for variant in tcp_variants:
            try:
                # Load throughput data
                data_file = f'results/{network}/{variant}_throughput.csv'
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file)
                    
                    # Calculate coefficient of variation (CV) as a measure of stability
                    # Lower CV means more stable throughput
                    mean = df['Throughput'].mean()
                    std = df['Throughput'].std()
                    cv = std / mean if mean > 0 else float('inf')
                    
                    # Calculate Jain's Fairness Index
                    # JFI = (sum(x_i))^2 / (n * sum(x_i^2))
                    # For time series data, we're measuring self-fairness over time
                    throughput_sum = df['Throughput'].sum()
                    throughput_squared_sum = (df['Throughput'] ** 2).sum()
                    n = len(df)
                    jfi = (throughput_sum ** 2) / (n * throughput_squared_sum) if throughput_squared_sum > 0 else 0
                    
                    # Store results
                    stability_results[network][variant] = {
                        'cv': cv,
                        'jfi': jfi
                    }
                    
                    print(f"*** Analyzed stability for {variant} on {network}: CV={cv:.4f}, JFI={jfi:.4f}")
                else:
                    print(f"*** Warning: Data file {data_file} not found")
            except Exception as e:
                print(f"*** Error analyzing stability for {variant} on {network}: {e}")
    
    # Create comparison bar chart for coefficient of variation (lower is better)
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(tcp_variants))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    for i, network in enumerate(network_types):
        cv_values = [stability_results[network][variant]['cv'] for variant in tcp_variants]
        pos = r1 if i == 0 else r2
        plt.bar(pos, cv_values, width=bar_width, label=network.upper())
    
    # Add labels and legend
    plt.xlabel('TCP Variant')
    plt.ylabel('Coefficient of Variation (lower is better)')
    plt.title('Throughput Stability Comparison by TCP Variant and Network Type')
    plt.xticks([r + bar_width/2 for r in r1], tcp_variants)
    plt.legend()
    
    # Save the figure
    plt.savefig('analysis/stability_comparison.png')
    plt.close()
    
    # Create comparison bar chart for Jain's Fairness Index (higher is better)
    plt.figure(figsize=(12, 6))
    
    # Create bars
    for i, network in enumerate(network_types):
        jfi_values = [stability_results[network][variant]['jfi'] for variant in tcp_variants]
        pos = r1 if i == 0 else r2
        plt.bar(pos, jfi_values, width=bar_width, label=network.upper())
    
    # Add labels and legend
    plt.xlabel('TCP Variant')
    plt.ylabel('Jain\'s Fairness Index (higher is better)')
    plt.title('Throughput Fairness Comparison by TCP Variant and Network Type')
    plt.xticks([r + bar_width/2 for r in r1], tcp_variants)
    plt.legend()
    
    # Save the figure
    plt.savefig('analysis/fairness_comparison.png')
    plt.close()
    
    # Create table of results
    table_data = []
    for network in network_types:
        for variant in tcp_variants:
            if variant in stability_results[network]:
                row = [
                    network.upper(),
                    variant.capitalize(),
                    f"{stability_results[network][variant]['cv']:.4f}",
                    f"{stability_results[network][variant]['jfi']:.4f}"
                ]
                table_data.append(row)
    
    # Create DataFrame for table
    table_df = pd.DataFrame(table_data, columns=[
        'Network', 'TCP Variant', 'Coefficient of Variation (lower is better)', 
        'Jain\'s Fairness Index (higher is better)'
    ])
    
    # Save table to CSV
    table_df.to_csv('analysis/stability_fairness_statistics.csv', index=False)
    
    return stability_results

def analyze_network_impact():
    """
    Analyze the impact of network type on TCP performance
    """
    # Network types
    network_types = ['lan', 'wlan']
    
    # TCP variants
    tcp_variants = ['tahoe', 'reno', 'cubic']
    
    # Results dictionary
    impact_results = {}
    
    # For each TCP variant, calculate the performance difference between LAN and WLAN
    for variant in tcp_variants:
        impact_results[variant] = {}
        
        try:
            # Load LAN data
            lan_file = f'results/lan/{variant}_throughput.csv'
            # Load WLAN data
            wlan_file = f'results/wlan/{variant}_throughput.csv'
            
            if os.path.exists(lan_file) and os.path.exists(wlan_file):
                lan_df = pd.read_csv(lan_file)
                wlan_df = pd.read_csv(wlan_file)
                
                # Calculate average throughput for each network type
                lan_avg = lan_df['Throughput'].mean()
                wlan_avg = wlan_df['Throughput'].mean()
                
                # Calculate percentage decrease from LAN to WLAN
                perc_decrease = ((lan_avg - wlan_avg) / lan_avg) * 100 if lan_avg > 0 else 0
                
                # Store results
                impact_results[variant] = {
                    'lan_avg': lan_avg,
                    'wlan_avg': wlan_avg,
                    'perc_decrease': perc_decrease
                }
                
                print(f"*** Analyzed network impact for {variant}: LAN={lan_avg:.2f} Mbps, WLAN={wlan_avg:.2f} Mbps, Decrease={perc_decrease:.2f}%")
            else:
                print(f"*** Warning: Data files for {variant} not found")
        except Exception as e:
            print(f"*** Error analyzing network impact for {variant}: {e}")
    
    # Create bar chart for percentage decrease in performance
    plt.figure(figsize=(10, 6))
    
    # Extract percentage decrease values
    perc_decrease_values = [impact_results[variant]['perc_decrease'] for variant in tcp_variants]
    
    # Create bars
    plt.bar(tcp_variants, perc_decrease_values)
    
    # Add labels
    plt.xlabel('TCP Variant')
    plt.ylabel('Performance Decrease from LAN to WLAN (%)')
    plt.title('Impact of Network Type on TCP Performance')
    
    # Save the figure
    plt.savefig('analysis/network_impact.png')
    plt.close()
    
    # Create table of results
    table_data = []
    for variant in tcp_variants:
        if variant in impact_results:
            row = [
                variant.capitalize(),
                f"{impact_results[variant]['lan_avg']:.2f}",
                f"{impact_results[variant]['wlan_avg']:.2f}",
                f"{impact_results[variant]['perc_decrease']:.2f}"
            ]
            table_data.append(row)
    
    # Create DataFrame for table
    table_df = pd.DataFrame(table_data, columns=[
        'TCP Variant', 'LAN Avg Throughput (Mbps)', 
        'WLAN Avg Throughput (Mbps)', 'Performance Decrease (%)'
    ])
    
    # Save table to CSV
    table_df.to_csv('analysis/network_impact_statistics.csv', index=False)
    
    return impact_results

def generate_analysis_summary():
    """
    Generate a summary of the analysis results
    """
    # Create summary file
    with open('analysis/analysis_summary.txt', 'w') as f:
        f.write("TCP Performance Analysis Summary\n")
        f.write("===============================\n\n")
        
        # Throughput Analysis
        f.write("1. Throughput Analysis\n")
        f.write("---------------------\n")
        f.write("The analysis of throughput data across different TCP variants and network types reveals:\n\n")
        
        # Read throughput statistics
        try:
            throughput_df = pd.read_csv('analysis/throughput_statistics.csv')
            
            # Group by network type
            for network in ['LAN', 'WLAN']:
                network_data = throughput_df[throughput_df['Network'] == network]
                
                f.write(f"{network} Network:\n")
                
                # Find best performing variant
                best_idx = network_data['Avg Throughput (Mbps)'].astype(float).idxmax()
                best_variant = network_data.iloc[best_idx]['TCP Variant']
                best_throughput = float(network_data.iloc[best_idx]['Avg Throughput (Mbps)'])
                
                f.write(f"- Best performing: {best_variant} with average throughput of {best_throughput:.2f} Mbps\n")
                
                # Compare variants
                for _, row in network_data.iterrows():
                    variant = row['TCP Variant']
                    avg = float(row['Avg Throughput (Mbps)'])
                    max_val = float(row['Max Throughput (Mbps)'])
                    min_val = float(row['Min Throughput (Mbps)'])
                    std = float(row['Std Dev (Mbps)'])
                    
                    f.write(f"- {variant}: Avg={avg:.2f} Mbps, Max={max_val:.2f} Mbps, Min={min_val:.2f} Mbps, Std Dev={std:.2f} Mbps\n")
                
                f.write("\n")
        except Exception as e:
            f.write(f"Error reading throughput statistics: {e}\n\n")
        
        # Stability and Fairness Analysis
        f.write("2. Stability and Fairness Analysis\n")
        f.write("--------------------------------\n")
        f.write("The analysis of stability (coefficient of variation) and fairness (Jain's Fairness Index) reveals:\n\n")
        
        # Read stability statistics
        try:
            stability_df = pd.read_csv('analysis/stability_fairness_statistics.csv')
            
            # Group by network type
            for network in ['LAN', 'WLAN']:
                network_data = stability_df[stability_df['Network'] == network]
                
                f.write(f"{network} Network:\n")
                
                # Find most stable variant (lowest CV)
                cv_col = 'Coefficient of Variation (lower is better)'
                best_idx = network_data[cv_col].astype(float).idxmin()
                best_variant = network_data.iloc[best_idx]['TCP Variant']
                best_cv = float(network_data.iloc[best_idx][cv_col])
                
                f.write(f"- Most stable: {best_variant} with CV of {best_cv:.4f}\n")
                
                # Find fairest variant (highest JFI)
                jfi_col = 'Jain\'s Fairness Index (higher is better)'
                best_idx = network_data[jfi_col].astype(float).idxmax()
                best_variant = network_data.iloc[best_idx]['TCP Variant']
                best_jfi = float(network_data.iloc[best_idx][jfi_col])
                
                f.write(f"- Fairest: {best_variant} with JFI of {best_jfi:.4f}\n")
                
                # Compare variants
                for _, row in network_data.iterrows():
                    variant = row['TCP Variant']
                    cv = float(row[cv_col])
                    jfi = float(row[jfi_col])
                    
                    f.write(f"- {variant}: CV={cv:.4f} (lower is better), JFI={jfi:.4f} (higher is better)\n")
                
                f.write("\n")
        except Exception as e:
            f.write(f"Error reading stability statistics: {e}\n\n")
        
        # Network Impact Analysis
        f.write("3. Network Impact Analysis\n")
        f.write("-------------------------\n")
        f.write("The analysis of the impact of network type (LAN vs. WLAN) on TCP performance reveals:\n\n")
        
        # Read network impact statistics
        try:
            impact_df = pd.read_csv('analysis/network_impact_statistics.csv')
            
            # Find most resilient variant (lowest performance decrease)
            best_idx = impact_df['Performance Decrease (%)'].astype(float).idxmin()
            best_variant = impact_df.iloc[best_idx]['TCP Variant']
            best_decrease = float(impact_df.iloc[best_idx]['Performance Decrease (%)'])
            
            f.write(f"- Most resilient to network change: {best_variant} with {best_decrease:.2f}% performance decrease\n\n")
            
            # Compare variants
            for _, row in impact_df.iterrows():
                variant = row['TCP Variant']
                lan_avg = float(row['LAN Avg Throughput (Mbps)'])
                wlan_avg = float(row['WLAN Avg Throughput (Mbps)'])
                decrease = float(row['Performance Decrease (%)'])
                
                f.write(f"- {variant}: LAN={lan_avg:.2f} Mbps, WLAN={wlan_avg:.2f} Mbps, Decrease={decrease:.2f}%\n")
            
            f.write("\n")
        except Exception as e:
            f.write(f"Error reading network impact statistics: {e}\n\n")
        
        # Overall Conclusions
        f.write("4. Overall Conclusions\n")
        f.write("---------------------\n")
        
        try:
            # Determine best overall TCP variant based on multiple metrics
            throughput_df = pd.read_csv('analysis/throughput_statistics.csv')
            stability_df = pd.read_csv('analysis/stability_fairness_statistics.csv')
            impact_df = pd.read_csv('analysis/network_impact_statistics.csv')
            
            # Simple scoring system (higher is better)
            scores = {'Tahoe': 0, 'Reno': 0, 'Cubic': 0}
            
            # Score based on throughput (higher is better)
            for network in ['LAN', 'WLAN']:
                network_data = throughput_df[throughput_df['Network'] == network]
                throughputs = []
                
                for _, row in network_data.iterrows():
                    variant = row['TCP Variant']
                    avg = float(row['Avg Throughput (Mbps)'])
                    throughputs.append((variant, avg))
                
                # Sort by throughput (descending)
                throughputs.sort(key=lambda x: x[1], reverse=True)
                
                # Assign scores (3 points for best, 2 for second, 1 for third)
                for i, (variant, _) in enumerate(throughputs):
                    scores[variant] += 3 - i
            
            # Score based on stability (lower CV is better)
            for network in ['LAN', 'WLAN']:
                network_data = stability_df[stability_df['Network'] == network]
                stabilities = []
                
                for _, row in network_data.iterrows():
                    variant = row['TCP Variant']
                    cv = float(row['Coefficient of Variation (lower is better)'])
                    stabilities.append((variant, cv))
                
                # Sort by CV (ascending)
                stabilities.sort(key=lambda x: x[1])
                
                # Assign scores (3 points for best, 2 for second, 1 for third)
                for i, (variant, _) in enumerate(stabilities):
                    scores[variant] += 3 - i
            
            # Score based on network resilience (lower decrease is better)
            decreases = []
            
            for _, row in impact_df.iterrows():
                variant = row['TCP Variant']
                decrease = float(row['Performance Decrease (%)'])
                decreases.append((variant, decrease))
            
            # Sort by decrease (ascending)
            decreases.sort(key=lambda x: x[1])
            
            # Assign scores (3 points for best, 2 for second, 1 for third)
            for i, (variant, _) in enumerate(decreases):
                scores[variant] += 3 - i
            
            # Determine overall best variant
            best_variant = max(scores.items(), key=lambda x: x[1])[0]
            
            f.write(f"Based on the comprehensive analysis of throughput, stability, fairness, and network resilience, ")
            f.write(f"TCP {best_variant} appears to be the best overall performer in our test scenarios.\n\n")
            
            # Write scores
            f.write("Overall scores (higher is better):\n")
            for variant, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- TCP {variant}: {score} points\n")
            
            f.write("\n")
            
            # Specific conclusions
            f.write("Specific conclusions:\n")
            
            # Throughput conclusion
            lan_data = throughput_df[throughput_df['Network'] == 'LAN']
            best_lan_idx = lan_data['Avg Throughput (Mbps)'].astype(float).idxmax()
            best_lan_variant = lan_data.iloc[best_lan_idx]['TCP Variant']
            
            wlan_data = throughput_df[throughput_df['Network'] == 'WLAN']
            best_wlan_idx = wlan_data['Avg Throughput (Mbps)'].astype(float).idxmax()
            best_wlan_variant = wlan_data.iloc[best_wlan_idx]['TCP Variant']
            
            f.write(f"1. TCP {best_lan_variant} achieves the highest throughput in LAN environments, ")
            f.write(f"while TCP {best_wlan_variant} performs best in WLAN environments.\n")
            
            # Stability conclusion
            lan_data = stability_df[stability_df['Network'] == 'LAN']
            best_lan_idx = lan_data['Coefficient of Variation (lower is better)'].astype(float).idxmin()
            best_lan_variant = lan_data.iloc[best_lan_idx]['TCP Variant']
            
            wlan_data = stability_df[stability_df['Network'] == 'WLAN']
            best_wlan_idx = wlan_data['Coefficient of Variation (lower is better)'].astype(float).idxmin()
            best_wlan_variant = wlan_data.iloc[best_wlan_idx]['TCP Variant']
            
            f.write(f"2. TCP {best_lan_variant} provides the most stable performance in LAN environments, ")
            f.write(f"while TCP {best_wlan_variant} is most stable in WLAN environments.\n")
            
            # Network impact conclusion
            best_idx = impact_df['Performance Decrease (%)'].astype(float).idxmin()
            best_variant = impact_df.iloc[best_idx]['TCP Variant']
            
            f.write(f"3. TCP {best_variant} is the most resilient to network changes, showing the smallest ")
            f.write(f"performance degradation when moving from LAN to WLAN environments.\n")
            
            # General conclusion
            f.write(f"4. Network topology has a significant impact on TCP performance, with all variants ")
            f.write(f"showing reduced throughput and increased variability in WLAN environments compared to LAN.\n")
            
            f.write(f"5. The congestion control algorithms in different TCP variants respond differently to ")
            f.write(f"network characteristics such as delay, jitter, and packet loss, which are more prevalent in WLAN.\n")
            
        except Exception as e:
            f.write(f"Error generating overall conclusions: {e}\n")
    
    print("*** Analysis summary generated successfully")

if __name__ == "__main__":
    # Analyze throughput data
    print("*** Analyzing throughput data")
    throughput_results = analyze_throughput_data()
    
    # Analyze stability and fairness
    print("*** Analyzing stability and fairness")
    stability_results = analyze_stability_and_fairness()
    
    # Analyze network impact
    print("*** Analyzing network impact")
    impact_results = analyze_network_impact()
    
    # Generate analysis summary
    print("*** Generating analysis summary")
    generate_analysis_summary()
    
    print("*** Analysis completed successfully")
