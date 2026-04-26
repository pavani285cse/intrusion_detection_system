"""
EGNNN-GROA-BGPoW Real-Time IDS
================================
Live network packet capture → feature extraction → 
EGNNN classification → blockchain logging → terminal dashboard

Requirements:
  pip install scapy psutil colorama
  Windows: install Npcap from https://npcap.com/#download

Run:
  python realtime_ids.py
  (may need admin/root privileges for packet capture)
"""

import time
import threading
import collections
import hashlib
import json
import numpy as np
import torch
import psutil
from datetime import datetime
from colorama import Fore, Style, init as colorama_init

# Scapy imports
from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list

# Your existing modules
from stage3_egnnn    import EGNNNClassifier
from stage5_blockchain import BGPoWBlockchain

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = "egnnn_best_weights.pt"
INPUT_DIM          = 18          # match your trained model (18 or 20)
TIME_WINDOW        = 2.0         # seconds — sliding window for rate features
MAX_CONNECTIONS    = 500         # max connections tracked in memory
BLOCKCHAIN_CAP     = 500         # max blocks to mine in one session
PACKET_TIMEOUT     = 0.1         # scapy sniff timeout per batch

# Attack label colors for terminal
LABEL_COLORS = {
    "Normal": Fore.GREEN,
    "DoS":    Fore.RED,
    "Probe":  Fore.YELLOW,
    "R2L":    Fore.MAGENTA,
    "U2R":    Fore.CYAN,
}

# Class index → label (must match your label encoder order)
# Order: alphabetical as fitted by LabelEncoder
# ['DoS', 'Normal', 'Probe', 'R2L', 'U2R'] → indices 0,1,2,3,4
IDX_TO_LABEL = {0: "DoS", 1: "Normal", 2: "Probe", 3: "R2L", 4: "U2R"}

# Protocol mapping → integer (matches NSL-KDD encoding)
PROTOCOL_MAP = {"icmp": 0, "tcp": 1, "udp": 2}

# Common service port → integer code (simplified NSL-KDD service mapping)
PORT_SERVICE_MAP = {
    80: 10, 443: 10, 21: 5, 22: 6, 23: 7, 25: 8,
    53: 9,  110: 11, 143: 12, 3306: 13, 8080: 10,
    0: 0
}

# TCP flag mapping → integer (NSL-KDD flag encoding simplified)
TCP_FLAG_MAP = {
    0x02: 1,   # SYN
    0x12: 2,   # SYN-ACK
    0x10: 3,   # ACK
    0x18: 4,   # PSH-ACK
    0x01: 5,   # FIN
    0x04: 6,   # RST
    0x11: 7,   # FIN-ACK
    0x14: 8,   # RST-ACK
}


# ─────────────────────────────────────────────────────────────
# CONNECTION TRACKER — sliding window over recent packets
# ─────────────────────────────────────────────────────────────
class ConnectionTracker:
    """
    Maintains a sliding window of recent connections.
    Used to compute rate-based NSL-KDD features like:
      count, same_srv_rate, diff_srv_rate, serror_rate etc.
    """

    def __init__(self, window_seconds: float = 2.0, maxlen: int = 500):
        self.window  = window_seconds
        self.records = collections.deque(maxlen=maxlen)
        self.lock    = threading.Lock()

    def add(self, record: dict):
        record["timestamp"] = time.time()
        with self.lock:
            self.records.append(record)

    def get_window(self):
        """Return records within the last TIME_WINDOW seconds."""
        cutoff = time.time() - self.window
        with self.lock:
            return [r for r in self.records if r["timestamp"] >= cutoff]

    def compute_rates(self, dst_ip: str, dst_port: int) -> dict:
        """
        Compute count-based features from the sliding window.
        Mirrors NSL-KDD connection-level rate features.
        """
        window_records = self.get_window()
        total = len(window_records)

        if total == 0:
            return {
                "count": 0, "same_srv_rate": 0.0,
                "diff_srv_rate": 0.0, "dst_host_count": 0,
                "dst_host_same_src_port_rate": 0.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_diff_host_rate": 0.0,
            }

        # Connections to same destination IP
        same_dst   = [r for r in window_records if r.get("dst_ip") == dst_ip]
        # Connections to same service (port)
        same_srv   = [r for r in same_dst if r.get("dst_port") == dst_port]
        diff_srv   = [r for r in same_dst if r.get("dst_port") != dst_port]

        # SYN errors (SYN sent, no response)
        syn_errors = [r for r in same_dst if r.get("flag") == 1]
        # RST responses
        rst_errors = [r for r in same_dst if r.get("flag") in (6, 8)]

        n_dst = len(same_dst) or 1

        # Unique source IPs hitting same dst service
        unique_src = len(set(r.get("src_ip", "") for r in same_srv))
        n_same_srv = len(same_srv) or 1

        return {
            "count":                      min(total, 511),
            "same_srv_rate":              len(same_srv) / n_dst,
            "diff_srv_rate":              len(diff_srv) / n_dst,
            "dst_host_count":             min(len(same_dst), 255),
            "dst_host_same_src_port_rate": len(same_srv) / n_dst,
            "dst_host_diff_srv_rate":     len(diff_srv) / n_dst,
            "dst_host_serror_rate":       len(syn_errors) / n_dst,
            "dst_host_rerror_rate":       len(rst_errors) / n_dst,
            "dst_host_srv_diff_host_rate": unique_src / n_same_srv,
        }


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────
class FeatureExtractor:
    """
    Converts a raw scapy packet into the 18-feature vector
    your EGNNN model was trained on.

    Feature order (must match stage2 final_features):
      same_srv_rate, diff_srv_rate, logged_in, protocol_type,
      hot, count, dst_host_same_src_port_rate,
      dst_host_srv_diff_host_rate, dst_host_serror_rate,
      dst_host_rerror_rate, dst_host_diff_srv_rate,
      src_bytes, dst_bytes, num_root, service, land, flag,
      wrong_fragment
    """

    def __init__(self, tracker: ConnectionTracker, scaler=None):
        self.tracker = tracker
        self.scaler  = scaler   # optional: pass MinMaxScaler from Stage 1

    def extract(self, packet) -> np.ndarray | None:
        """
        Extract features from a single packet.
        Returns float32 array of shape (18,) or None if packet not usable.
        """
        if not packet.haslayer(IP):
            return None

        ip    = packet[IP]
        src_ip  = ip.src
        dst_ip  = ip.dst
        src_bytes = len(packet)   # total packet size as proxy for src_bytes

        # ── Protocol ─────────────────────────────────────────
        if packet.haslayer(TCP):
            proto    = PROTOCOL_MAP["tcp"]
            dst_port = packet[TCP].dport
            src_port = packet[TCP].sport
            tcp_flags = int(packet[TCP].flags)
            flag_code = TCP_FLAG_MAP.get(tcp_flags & 0x3F, 0)
            dst_bytes = 0   # can't know response size from single packet
        elif packet.haslayer(UDP):
            proto    = PROTOCOL_MAP["udp"]
            dst_port = packet[UDP].dport
            src_port = packet[UDP].sport
            flag_code = 0
            dst_bytes = 0
        elif packet.haslayer(ICMP):
            proto    = PROTOCOL_MAP["icmp"]
            dst_port = 0
            src_port = 0
            flag_code = 0
            dst_bytes = 0
        else:
            return None

        # ── Service (port → code) ─────────────────────────────
        service_code = PORT_SERVICE_MAP.get(dst_port, 1)

        # ── Land (src_ip:port == dst_ip:port → attack indicator)
        land = 1 if (src_ip == dst_ip and src_port == dst_port) else 0

        # ── Rate features from sliding window ─────────────────
        record = {
            "src_ip": src_ip, "dst_ip": dst_ip,
            "dst_port": dst_port, "flag": flag_code,
        }
        self.tracker.add(record)
        rates = self.tracker.compute_rates(dst_ip, dst_port)

        # ── Assemble feature vector (18 features) ─────────────
        # Order must match your model's training feature order
        features = np.array([
            rates["same_srv_rate"],               # 0
            rates["diff_srv_rate"],               # 1
            0.0,                                  # 2  logged_in (unknown from packet)
            float(proto),                         # 3  protocol_type
            0.0,                                  # 4  hot (unknown from packet)
            float(rates["count"]),                # 5  count
            rates["dst_host_same_src_port_rate"], # 6
            rates["dst_host_srv_diff_host_rate"], # 7
            rates["dst_host_serror_rate"],        # 8
            rates["dst_host_rerror_rate"],        # 9
            rates["dst_host_diff_srv_rate"],      # 10
            float(min(src_bytes, 1e9)),           # 11 src_bytes
            float(dst_bytes),                     # 12 dst_bytes
            0.0,                                  # 13 num_root (unknown)
            float(service_code),                  # 14 service
            float(land),                          # 15 land
            float(flag_code),                     # 16 flag
            0.0,                                  # 17 wrong_fragment (unknown)
        ], dtype=np.float32)

        # ── Normalize to [0,1] using same ranges as training ──
        # Simple clipping normalization (scaler not available at runtime)
        # src_bytes: clip at 1e9, count: clip at 511
        features[5]  = features[5]  / 511.0    # count
        features[11] = np.clip(features[11] / 1e6, 0, 1)  # src_bytes

        return features


# ─────────────────────────────────────────────────────────────
# LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
class LiveDashboard:
    """
    Prints color-coded real-time alerts to terminal.
    Tracks statistics: total packets, alerts per class, blockchain blocks.
    """

    def __init__(self):
        self.total_packets  = 0
        self.total_alerts   = 0
        self.class_counts   = collections.defaultdict(int)
        self.start_time     = time.time()
        self.lock           = threading.Lock()

    def update(self, label: str, confidence: float,
               src_ip: str, dst_ip: str, proto: str,
               is_attack: bool, block_added: bool):

        with self.lock:
            self.total_packets += 1
            self.class_counts[label] += 1
            if is_attack:
                self.total_alerts += 1

        color    = LABEL_COLORS.get(label, Fore.WHITE)
        ts       = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        chain_marker = f"{Fore.CYAN}[BLOCK ADDED]" if block_added else ""

        if is_attack:
            print(
                f"{Fore.WHITE}[{ts}] "
                f"{color}⚠  {label:<8}{Style.RESET_ALL} "
                f"conf={confidence:.2f} | "
                f"{src_ip} → {dst_ip} "
                f"({proto}) "
                f"{chain_marker}"
            )
        # Normal traffic is silent (don't flood terminal)

    def print_stats(self):
        elapsed = time.time() - self.start_time
        pps     = self.total_packets / max(elapsed, 1)
        print(f"\n{'─'*55}")
        print(f"  {Fore.WHITE}LIVE IDS STATISTICS")
        print(f"{'─'*55}")
        print(f"  Uptime        : {elapsed:.0f}s")
        print(f"  Packets seen  : {self.total_packets}")
        print(f"  Packets/sec   : {pps:.1f}")
        print(f"  Total alerts  : {self.total_alerts}")
        print(f"\n  Detections by class:")
        for label, count in sorted(self.class_counts.items()):
            color = LABEL_COLORS.get(label, Fore.WHITE)
            bar   = "█" * min(count // 5 + 1, 30)
            print(f"    {color}{label:<10}{Style.RESET_ALL} {count:>6}  {bar}")
        print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────
# REAL-TIME IDS ENGINE
# ─────────────────────────────────────────────────────────────
class RealTimeIDS:
    """
    Main engine. Loads trained EGNNN model, captures live packets,
    extracts features, classifies each packet, logs attacks to blockchain.
    """

    def __init__(self,
                 model_path:  str   = MODEL_WEIGHTS_PATH,
                 input_dim:   int   = INPUT_DIM,
                 interface:   str   = None,
                 packet_count: int  = 0):      # 0 = infinite

        self.input_dim    = input_dim
        self.interface    = interface
        self.packet_count = packet_count
        self.blocks_added = 0

        # Load model
        print(f"[IDS] Loading EGNNN model from '{model_path}'...")
        self.model = EGNNNClassifier(
            input_dim=input_dim,
            layer_sizes=[64, 128, 256, 128, 64],
            n_classes=5
        )
        self.model.load_weights(model_path)
        self.model.eval_mode()
        print(f"[IDS] Model loaded. Input dim: {input_dim}")

        # Blockchain
        self.blockchain = BGPoWBlockchain(n_nodes=5)

        # Tracker, extractor, dashboard
        self.tracker   = ConnectionTracker(window_seconds=TIME_WINDOW)
        self.extractor = FeatureExtractor(self.tracker)
        self.dashboard = LiveDashboard()

        # Stats thread — prints summary every 30s
        self._stop_event = threading.Event()
        self._stats_thread = threading.Thread(
            target=self._stats_loop, daemon=True
        )

    def _stats_loop(self):
        while not self._stop_event.is_set():
            time.sleep(30)
            self.dashboard.print_stats()

    def process_packet(self, packet):
        """Called by scapy for every captured packet."""
        features = self.extractor.extract(packet)
        if features is None:
            return

        # Classify
        features_tensor = torch.tensor(features).unsqueeze(0)  # (1, 18)
        label, confidence, proba = self.model.predict_single(features_tensor)
        label_str = IDX_TO_LABEL.get(label, "Unknown")
        # Only alert if confidence is above 60% threshold
        is_attack = label_str != "Normal" and confidence > 0.60

        # Determine protocol string
        if packet.haslayer(TCP):   proto = "TCP"
        elif packet.haslayer(UDP): proto = "UDP"
        else:                      proto = "ICMP"

        src_ip = packet[IP].src if packet.haslayer(IP) else "?"
        dst_ip = packet[IP].dst if packet.haslayer(IP) else "?"

        # Blockchain logging for attacks
        block_added = False
        if is_attack and self.blocks_added < BLOCKCHAIN_CAP:
            self.blockchain.add_block({
                "sample_id":       self.dashboard.total_packets,
                "predicted_label": label_str,
                "true_label":      "unknown",   # real-time: no ground truth
                "confidence":      float(confidence),
                "is_intrusion":    True,
                "src_ip":          src_ip,
                "dst_ip":          dst_ip,
                "protocol":        proto,
                "timestamp":       datetime.now().isoformat(),
            })
            self.blocks_added += 1
            block_added = True

        # Update dashboard
        self.dashboard.update(
            label_str, confidence, src_ip, dst_ip,
            proto, is_attack, block_added
        )

    def start(self):
        """Start live packet capture."""
        self._stats_thread.start()

        print(f"\n{'═'*55}")
        print(f"  {Fore.GREEN}EGNNN-GROA REAL-TIME IDS — STARTED{Style.RESET_ALL}")
        print(f"{'═'*55}")
        print(f"  Interface : {self.interface or 'auto (default)'}")
        print(f"  Window    : {TIME_WINDOW}s sliding")
        print(f"  Press Ctrl+C to stop\n")
        print(f"  {'Time':12} {'Class':10} {'Conf':6}  {'Flow':30}")
        print(f"  {'─'*12} {'─'*10} {'─'*6}  {'─'*30}")

        try:
            sniff(
                iface=self.interface,
                prn=self.process_packet,
                store=False,                # don't store packets in memory
                count=self.packet_count,    # 0 = infinite
                filter="ip",               # only capture IP packets
            )
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}[IDS] Stopping capture...{Style.RESET_ALL}")
        finally:
            self._stop_event.set()
            self._print_final_summary()

    def _print_final_summary(self):
        """Print final stats and blockchain summary on exit."""
        print(f"\n{'═'*55}")
        print(f"  {Fore.GREEN}SESSION COMPLETE{Style.RESET_ALL}")
        print(f"{'═'*55}")
        self.dashboard.print_stats()

        print(f"\n  {Fore.CYAN}BLOCKCHAIN SUMMARY{Style.RESET_ALL}")
        self.blockchain.validate_chain()
        self.blockchain.print_chain_summary()


# ─────────────────────────────────────────────────────────────
# NETWORK INTERFACE SELECTOR
# ─────────────────────────────────────────────────────────────
def select_interface() -> str:
    """
    List available network interfaces and let user pick one.
    On Windows with Npcap, scapy can see all adapters.
    """
    interfaces = get_if_list()
    print(f"\n{Fore.WHITE}Available network interfaces:{Style.RESET_ALL}")
    for i, iface in enumerate(interfaces):
        # Try to get stats to show active ones
        try:
            stats = psutil.net_io_counters(pernic=True).get(iface)
            activity = f"  ↑{stats.bytes_sent/1e6:.1f}MB ↓{stats.bytes_recv/1e6:.1f}MB" if stats else ""
        except Exception:
            activity = ""
        print(f"  [{i}] {iface}{activity}")

    print(f"\n  [Enter] Use default interface")
    choice = input("Select interface number: ").strip()

    if choice == "":
        return None   # scapy picks default
    try:
        return interfaces[int(choice)]
    except (ValueError, IndexError):
        print("[IDS] Invalid choice — using default interface.")
        return None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print(f"{Fore.CYAN}")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║   EGNNN-GROA-BGPoW Real-Time IDS v1.0   ║")
    print("  ║   Live Network Intrusion Detection       ║")
    print("  ╚══════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    # Check model file exists
    import os
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"{Fore.RED}[ERROR] Model weights not found: '{MODEL_WEIGHTS_PATH}'")
        print(f"        Run main.py first to train and save the model.{Style.RESET_ALL}")
        sys.exit(1)

    # Select network interface
    iface = select_interface()

    # Start IDS
    ids = RealTimeIDS(
        model_path=MODEL_WEIGHTS_PATH,
        input_dim=INPUT_DIM,
        interface=iface,
        packet_count=0    # 0 = capture forever until Ctrl+C
    )
    ids.start()