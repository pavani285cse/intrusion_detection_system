"""
EGNNN-GROA-BGPoW Real-Time IDS v3.0 (Hybrid Detection Engine)
"""

import time
import threading
import collections
import numpy as np
import torch
import psutil
import requests
from datetime import datetime
from colorama import Fore, Style, init as colorama_init
from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
import ipaddress

from stage3_egnnn import EGNNNClassifier
from stage5_blockchain import BGPoWBlockchain

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
TESTING_MODE = False
MODEL_THRESHOLD = 0.55
CLOUD_API_URL = "http://localhost:5000/api/alert"
MODEL_WEIGHTS_PATH = "egnnn_best_weights.pt"
INPUT_DIM = 18
TIME_WINDOW = 2.0
BLOCKCHAIN_CAP = 500

IDX_TO_LABEL = {0: "DoS", 1: "Normal", 2: "Probe", 3: "R2L", 4: "U2R"}
PROTOCOL_MAP = {"icmp": 0, "tcp": 1, "udp": 2}
PORT_SERVICE_MAP = {80:10, 443:10, 21:5, 22:6, 23:7, 25:8, 53:9, 110:11, 143:12, 3306:13, 8080:10, 0:0}
TCP_FLAG_MAP = {0x02:1, 0x12:2, 0x10:3, 0x18:4, 0x01:5, 0x04:6, 0x11:7, 0x14:8}
LABEL_COLORS = {"Normal":Fore.GREEN, "DoS":Fore.RED, "Probe":Fore.YELLOW, "R2L":Fore.MAGENTA, "U2R":Fore.CYAN}

# ─────────────────────────────────────────────────────────────
# FREQUENCY TRACKER
# ─────────────────────────────────────────────────────────────
class FrequencyTracker:
    def __init__(self, window_seconds: float = 10.0):
        self.window = window_seconds
        self.records = collections.defaultdict(list)
        self.lock = threading.Lock()

    def add(self, src_ip, dst_ip, label):
        key = (src_ip, dst_ip, label)
        now = time.time()
        with self.lock:
            self.records[key].append(now)
            self.records[key] = [t for t in self.records[key] if now - t <= self.window]

    def is_frequent(self, src_ip, dst_ip, label) -> bool:
        key = (src_ip, dst_ip, label)
        now = time.time()
        with self.lock:
            recent = [t for t in self.records.get(key, []) if now - t <= self.window]
            return len(recent) >= 3

# ─────────────────────────────────────────────────────────────
# CONNECTION TRACKER
# ─────────────────────────────────────────────────────────────
class ConnectionTracker:
    def __init__(self, window_seconds=2.0, maxlen=500):
        self.window = window_seconds
        self.records = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def add(self, record):
        record["timestamp"] = time.time()
        with self.lock:
            self.records.append(record)

    def get_window(self):
        cutoff = time.time() - self.window
        with self.lock:
            return [r for r in self.records if r["timestamp"] >= cutoff]

    def compute_rates(self, dst_ip, dst_port):
        wr = self.get_window()
        total = len(wr)
        if total == 0:
            return {"count":0,"same_srv_rate":0.0,"diff_srv_rate":0.0,
                    "dst_host_count":0,"dst_host_same_src_port_rate":0.0,
                    "dst_host_diff_srv_rate":0.0,"dst_host_serror_rate":0.0,
                    "dst_host_rerror_rate":0.0,"dst_host_srv_diff_host_rate":0.0}

        same_dst = [r for r in wr if r.get("dst_ip") == dst_ip]
        same_srv = [r for r in same_dst if r.get("dst_port") == dst_port]
        diff_srv = [r for r in same_dst if r.get("dst_port") != dst_port]
        syn_errors = [r for r in same_dst if r.get("flag") == 1]
        rst_errors = [r for r in same_dst if r.get("flag") in (6,8)]
        n_dst = len(same_dst) or 1
        unique_src = len(set(r.get("src_ip","") for r in same_srv))
        n_same_srv = len(same_srv) or 1

        return {
            "count": min(total, 511),
            "same_srv_rate": len(same_srv) / n_dst,
            "diff_srv_rate": len(diff_srv) / n_dst,
            "dst_host_count": min(len(same_dst), 255),
            "dst_host_same_src_port_rate": len(same_srv) / n_dst,
            "dst_host_diff_srv_rate": len(diff_srv) / n_dst,
            "dst_host_serror_rate": len(syn_errors) / n_dst,
            "dst_host_rerror_rate": len(rst_errors) / n_dst,
            "dst_host_srv_diff_host_rate": unique_src / n_same_srv,
        }

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, tracker: ConnectionTracker):
        self.tracker = tracker

    def extract(self, packet):
        if not packet.haslayer(IP):
            return None

        ip = packet[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        src_bytes = len(packet)

        if packet.haslayer(TCP):
            proto = PROTOCOL_MAP["tcp"]
            dst_port = packet[TCP].dport
            src_port = packet[TCP].sport
            tcp_flags = int(packet[TCP].flags)
            flag_code = TCP_FLAG_MAP.get(tcp_flags & 0x3F, 0)
            dst_bytes = 0
        elif packet.haslayer(UDP):
            proto = PROTOCOL_MAP["udp"]
            dst_port = packet[UDP].dport
            src_port = packet[UDP].sport
            flag_code = 0; dst_bytes = 0
        elif packet.haslayer(ICMP):
            proto = PROTOCOL_MAP["icmp"]
            dst_port = 0; src_port = 0
            flag_code = 0; dst_bytes = 0
        else:
            return None

        service_code = PORT_SERVICE_MAP.get(dst_port, 1)
        land = 1 if (src_ip == dst_ip and src_port == dst_port) else 0

        record = {"src_ip": src_ip, "dst_ip": dst_ip, "dst_port": dst_port, "flag": flag_code}
        self.tracker.add(record)
        rates = self.tracker.compute_rates(dst_ip, dst_port)

        features = np.array([
            rates["same_srv_rate"], rates["diff_srv_rate"], 0.0, float(proto), 0.0,
            float(rates["count"]), rates["dst_host_same_src_port_rate"],
            rates["dst_host_srv_diff_host_rate"], rates["dst_host_serror_rate"],
            rates["dst_host_rerror_rate"], rates["dst_host_diff_srv_rate"],
            float(min(src_bytes, 1e9)), float(dst_bytes), 0.0, float(service_code),
            float(land), float(flag_code), 0.0
        ], dtype=np.float32)

        features[5] = features[5] / 511.0
        features[11] = np.clip(features[11] / 1e6, 0, 1)
        return features

# ─────────────────────────────────────────────────────────────
# LAYER 1 — RULE ENGINE
# ─────────────────────────────────────────────────────────────
class RuleEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.port_hits = collections.defaultdict(lambda: collections.defaultdict(list))
        self.syn_counts = collections.defaultdict(list)
        self.ssh_attempts = collections.defaultdict(lambda: collections.defaultdict(list))
        self.icmp_counts = collections.defaultdict(list)
        self.udp_counts = collections.defaultdict(list)

    def _clean_list(self, timestamps, window, now):
        while timestamps and now - timestamps[0] > window:
            timestamps.pop(0)

    def update(self, packet_info: dict) -> dict | None:
        src_ip = packet_info.get('src_ip')
        dst_port = packet_info.get('dst_port', 0)
        proto = packet_info.get('proto')
        tcp_flags = packet_info.get('tcp_flags', 0)
        now = packet_info.get('timestamp', time.time())

        with self.lock:
            for dst in list(self.port_hits[src_ip].keys()):
                self._clean_list(self.port_hits[src_ip][dst], 2.0, now)
                if not self.port_hits[src_ip][dst]:
                    del self.port_hits[src_ip][dst]
            
            self._clean_list(self.syn_counts[src_ip], 2.0, now)
            
            for port in list(self.ssh_attempts[src_ip].keys()):
                self._clean_list(self.ssh_attempts[src_ip][port], 5.0, now)
                if not self.ssh_attempts[src_ip][port]:
                    del self.ssh_attempts[src_ip][port]
                    
            self._clean_list(self.icmp_counts[src_ip], 2.0, now)
            self._clean_list(self.udp_counts[src_ip], 2.0, now)

            if proto == 'tcp':
                self.port_hits[src_ip][dst_port].append(now)
                if len(self.port_hits[src_ip]) >= 15:
                    return {
                        "rule_name": "PORT_SCAN",
                        "attack_type": "Probe",
                        "confidence": 1.0,
                        "severity": "HIGH",
                        "detail": f"{len(self.port_hits[src_ip])} ports scanned in 2.0s"
                    }
                
                if tcp_flags == 0x02:
                    self.syn_counts[src_ip].append(now)
                    if len(self.syn_counts[src_ip]) >= 100:
                        return {
                            "rule_name": "SYN_FLOOD",
                            "attack_type": "DoS",
                            "confidence": 1.0,
                            "severity": "HIGH",
                            "detail": f"{len(self.syn_counts[src_ip])} SYN packets in 2.0s"
                        }
                
                if dst_port in [21, 22, 23, 3389]:
                    self.ssh_attempts[src_ip][dst_port].append(now)
                    if len(self.ssh_attempts[src_ip][dst_port]) >= 20:
                        return {
                            "rule_name": "BRUTE_FORCE",
                            "attack_type": "R2L",
                            "confidence": 1.0,
                            "severity": "HIGH",
                            "detail": f"{len(self.ssh_attempts[src_ip][dst_port])} attempts to port {dst_port} in 5.0s"
                        }
            
            elif proto == 'icmp':
                self.icmp_counts[src_ip].append(now)
                if len(self.icmp_counts[src_ip]) >= 50:
                    return {
                        "rule_name": "ICMP_SWEEP",
                        "attack_type": "Probe",
                        "confidence": 1.0,
                        "severity": "MEDIUM",
                        "detail": f"{len(self.icmp_counts[src_ip])} ICMP packets in 2.0s"
                    }
            
            elif proto == 'udp':
                self.udp_counts[src_ip].append(now)
                if len(self.udp_counts[src_ip]) >= 150:
                    return {
                        "rule_name": "UDP_FLOOD",
                        "attack_type": "DoS",
                        "confidence": 1.0,
                        "severity": "MEDIUM",
                        "detail": f"{len(self.udp_counts[src_ip])} UDP packets in 2.0s"
                    }

        return None

# ─────────────────────────────────────────────────────────────
# LAYER 3 — CORRELATION ENGINE
# ─────────────────────────────────────────────────────────────
class CorrelationEngine:
    def correlate(self, rule_result, model_result) -> dict | None:
        if rule_result is not None and model_result is not None:
            return {
                "label": rule_result["attack_type"],
                "confidence": 0.95,
                "severity": "HIGH",
                "reason": "Rule + Model confirmed",
                "alert": True,
                "blockchain": True,
                "rule_detail": rule_result["detail"]
            }
        elif rule_result is not None and model_result is None:
            return {
                "label": rule_result["attack_type"],
                "confidence": 0.90,
                "severity": rule_result.get("severity", "HIGH"),
                "reason": rule_result["detail"],
                "alert": True,
                "blockchain": True,
                "rule_detail": rule_result["detail"]
            }
        elif rule_result is None and model_result is not None:
            return {
                "label": model_result["attack_type"],
                "confidence": model_result["confidence"],
                "severity": "SUSPICIOUS",
                "reason": "Model flagged, no rule triggered",
                "alert": False,
                "blockchain": False,
                "rule_detail": None
            }
        return None

# ─────────────────────────────────────────────────────────────
# LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
class LiveDashboard:
    def __init__(self):
        self.total_packets = 0
        self.total_alerts = 0
        self.class_counts = collections.defaultdict(int)
        self.severity_counts = {"HIGH": 0, "SUSPICIOUS": 0, "MEDIUM": 0}
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, final: dict, src_ip: str, dst_ip: str, proto: str):
        with self.lock:
            self.total_packets += 1
            if final is None:
                return
            
            label = final["label"]
            severity = final["severity"]
            self.class_counts[label] += 1
            if final["alert"]:
                self.total_alerts += 1
                
            if severity in self.severity_counts:
                self.severity_counts[severity] += 1
            else:
                self.severity_counts[severity] = 1

        color = Fore.RED if final["severity"] == "HIGH" else (Fore.YELLOW if final["severity"] == "SUSPICIOUS" else Fore.MAGENTA)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        chain_marker = f"{Fore.CYAN}[BLOCK ADDED]" if final.get("blockchain") else ""

        if final["alert"] or final["severity"] == "SUSPICIOUS":
            print(
                f"{Fore.WHITE}[{ts}] "
                f"{color}⚠  {severity:<10} | {label:<8}{Style.RESET_ALL} "
                f"conf={final['confidence']:.2f} | "
                f"{src_ip} → {dst_ip} ({proto}) "
                f"Reason: {final['reason']} "
                f"{chain_marker}"
            )

    def print_stats(self):
        elapsed = time.time() - self.start_time
        pps = self.total_packets / max(elapsed, 1)
        print(f"\n{'─'*65}")
        print(f"  {Fore.WHITE}LIVE IDS STATISTICS")
        print(f"{'─'*65}")
        print(f"  Uptime        : {elapsed:.0f}s")
        print(f"  Packets seen  : {self.total_packets}")
        print(f"  Packets/sec   : {pps:.1f}")
        print(f"  Total alerts  : {self.total_alerts}")
        print(f"\n  Severity breakdown:")
        print(f"    {Fore.RED}HIGH       : {self.severity_counts.get('HIGH', 0)}{Style.RESET_ALL}")
        print(f"    {Fore.YELLOW}SUSPICIOUS : {self.severity_counts.get('SUSPICIOUS', 0)}{Style.RESET_ALL}")
        print(f"    {Fore.MAGENTA}MEDIUM     : {self.severity_counts.get('MEDIUM', 0)}{Style.RESET_ALL}")
        print(f"\n  Detections by class:")
        for label, count in sorted(self.class_counts.items()):
            color = LABEL_COLORS.get(label, Fore.WHITE)
            print(f"    {color}{label:<10}{Style.RESET_ALL} {count:>6}")
        print(f"{'─'*65}\n")

# ─────────────────────────────────────────────────────────────
# REAL-TIME IDS ENGINE
# ─────────────────────────────────────────────────────────────
class RealTimeIDS:
    def __init__(self, model_path=MODEL_WEIGHTS_PATH, input_dim=INPUT_DIM, interface=None):
        self.interface = interface
        self.blocks_added = 0

        print(f"[IDS] Loading EGNNN model from '{model_path}'...")
        try:
            self.model = EGNNNClassifier(input_dim=input_dim, layer_sizes=[64, 128, 256, 128, 64], n_classes=5)
            self.model.load_weights(model_path)
            self.model.eval_mode()
        except Exception as e:
            print(f"[IDS] Error loading model: {e}")

        self.blockchain = BGPoWBlockchain(n_nodes=5)
        self.tracker = ConnectionTracker(window_seconds=TIME_WINDOW)
        self.extractor = FeatureExtractor(self.tracker)
        self.dashboard = LiveDashboard()
        
        self.rule_engine = RuleEngine()
        self.correlator = CorrelationEngine()

        self._stop_event = threading.Event()
        self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)

    def _stats_loop(self):
        while not self._stop_event.is_set():
            time.sleep(30)
            self.dashboard.print_stats()

    def classify_with_model(self, features):
        try:
            features_tensor = torch.tensor(features).unsqueeze(0)
            label, confidence, proba = self.model.predict_single(features_tensor)
            label_str = IDX_TO_LABEL.get(label, "Unknown")
            
            if label_str == "Normal" or confidence < MODEL_THRESHOLD:
                return None
            
            return {
                "attack_type": label_str,
                "confidence": float(confidence),
                "severity": "LOW"
            }
        except:
            return None

    def process_packet(self, packet):
        try:
            if not packet.haslayer(IP):
                return

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

            if packet.haslayer(TCP):
                proto = "tcp"
                dst_port = packet[TCP].dport
                tcp_flags = int(packet[TCP].flags)
            elif packet.haslayer(UDP):
                proto = "udp"
                dst_port = packet[UDP].dport
                tcp_flags = 0
            elif packet.haslayer(ICMP):
                proto = "icmp"
                dst_port = 0
                tcp_flags = 0
            else:
                return

            packet_info = {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "proto": proto,
                "tcp_flags": tcp_flags,
                "packet_size": len(packet),
                "timestamp": time.time()
            }

            features = self.extractor.extract(packet)
            
            rule_result = self.rule_engine.update(packet_info)
            model_result = self.classify_with_model(features) if features is not None else None
            
            final = self.correlator.correlate(rule_result, model_result)
            
            if final is None:
                self.dashboard.update(None, src_ip, dst_ip, proto.upper())
                return

            if final.get("blockchain") and self.blocks_added < BLOCKCHAIN_CAP:
                self.blockchain.add_block({
                    "sample_id": self.dashboard.total_packets,
                    "predicted_label": final["label"],
                    "true_label": "unknown",
                    "confidence": final["confidence"],
                    "is_intrusion": True,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto.upper(),
                    "timestamp": datetime.now().isoformat(),
                })
                self.blocks_added += 1

            if final.get("alert") or final["severity"] == "SUSPICIOUS":
                data = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto.upper(),
                    "attack_type": final["label"],
                    "confidence": final["confidence"],
                    "severity": final["severity"],
                    "rule_detail": final.get("rule_detail", "")
                }
                try:
                    requests.post(CLOUD_API_URL, json=data, timeout=1)
                except:
                    pass

            self.dashboard.update(final, src_ip, dst_ip, proto.upper())
        except Exception as e:
            pass

    def start(self):
        self._stats_thread.start()
        print(f"\n{'═'*65}")
        print(f"  {Fore.GREEN}EGNNN-GROA HYBRID IDS — STARTED{Style.RESET_ALL}")
        print(f"{'═'*65}")
        print(f"  Mode            : PRODUCTION")
        print(f"  Rules           : 5 active")
        print(f"  Model threshold : {MODEL_THRESHOLD}")
        print(f"  Cloud API       : {CLOUD_API_URL}")
        print(f"  Interface       : {self.interface or 'auto (default)'}")
        print(f"  Press Ctrl+C to stop\n")
        
        try:
            sniff(iface=self.interface, prn=self.process_packet, store=False, filter="ip")
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}[IDS] Stopping capture...{Style.RESET_ALL}")
        finally:
            self._stop_event.set()
            self._print_final_summary()

    def _print_final_summary(self):
        print(f"\n{'═'*65}")
        print(f"  {Fore.GREEN}SESSION COMPLETE{Style.RESET_ALL}")
        print(f"{'═'*65}")
        self.dashboard.print_stats()
        print(f"\n  {Fore.CYAN}BLOCKCHAIN SUMMARY{Style.RESET_ALL}")
        self.blockchain.validate_chain()
        self.blockchain.print_chain_summary()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    ids = RealTimeIDS()
    ids.start()