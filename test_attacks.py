import sys
import time
from scapy.all import IP, TCP, UDP, ICMP, send

def test_port_scan(target_ip):
    print(f"[*] Running PORT_SCAN test against {target_ip}...")
    for port in range(1, 16):
        pkt = IP(dst=target_ip)/TCP(dport=port, flags="S")
        send(pkt, verbose=False)

def test_syn_flood(target_ip):
    print(f"[*] Running SYN_FLOOD test against {target_ip}...")
    for _ in range(105):
        pkt = IP(dst=target_ip)/TCP(dport=80, flags="S")
        send(pkt, verbose=False)

def test_brute_force(target_ip):
    print(f"[*] Running BRUTE_FORCE test against {target_ip}...")
    for _ in range(25):
        pkt = IP(dst=target_ip)/TCP(dport=22, flags="PA")
        send(pkt, verbose=False)

def test_icmp_sweep(target_ip):
    print(f"[*] Running ICMP_SWEEP test against {target_ip}...")
    for i in range(1, 55):
        pkt = IP(dst=target_ip)/ICMP()
        send(pkt, verbose=False)

def test_udp_flood(target_ip):
    print(f"[*] Running UDP_FLOOD test against {target_ip}...")
    for _ in range(155):
        pkt = IP(dst=target_ip)/UDP(dport=53)
        send(pkt, verbose=False)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.1"
    print(f"Running all attack tests against {target}")
    print("Make sure realtime_ids.py is running first!")
    test_port_scan(target)
    time.sleep(3)
    test_syn_flood(target)
    time.sleep(3)
    test_brute_force(target)
    time.sleep(3)
    test_icmp_sweep(target)
    time.sleep(3)
    test_udp_flood(target)
    print("All tests complete! Check your IDS terminal.")
