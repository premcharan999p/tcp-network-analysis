# 🛰️ TCP Network Analysis and Cybersecurity Simulation

**Author:** [Prem Charan Namburu]  
**Language:** Python | **Framework:** Mininet | **Focus:** Network Performance & Cybersecurity Analytics  

## 🚀 Overview
This project simulates and analyzes the performance of TCP variants — **Tahoe, Reno, and Cubic** — in both **LAN and WLAN** environments.  
It measures **throughput, fairness, stability, and resilience**, generating analytical plots and reports.

## ⚙️ Features
- Automated Mininet-based LAN/WLAN network emulation  
- Python scripts for data generation and analysis  
- Metrics: Throughput, Fairness, Coefficient of Variation, and Network Impact  
- Visual comparison plots across TCP variants  
- Extensible for **Breach & Attack Simulation** or **AI-based traffic anomaly detection**

## 📊 Example Output
| TCP Variant | LAN (Mbps) | WLAN (Mbps) | Drop (%) |
|--------------|-------------|-------------|----------|
| Tahoe | 45.3 | 24.6 | 45.8% |
| Reno | 67.9 | 24.3 | 64.2% |
| Cubic | 81.4 | 26.0 | 68.0% |

**Insight:** Cubic performs best in LAN; Tahoe is most resilient under WLAN losses.

## 🧠 Cybersecurity Relevance
- Supports **attack simulation** and **threat detection modeling**  
- Demonstrates deep understanding of **network behavior under stress**  
- Aligns with **Security Engineering**, **BAS**, and **MXDR** systems used at NopalCyber

 ## How to Run
 python3 generate_data.py
 python3 analyze_results.py

## Technologies Used
- Python 3.10+
- Mininet
- Iperf
- Pandas
- Matplotlib
- NumPy
- Linux Networking Utilities

## 📈 Future Enhancements

🔹 Integrate Zeek / Suricata for IDS-driven anomaly detection
🔹 Extend topology to include simulated attack traffic (SYN floods, DDoS)
🔹 Introduce AI/ML models for throughput anomaly prediction
🔹 Build interactive dashboard using Streamlit or Plotly Dash for visualization
