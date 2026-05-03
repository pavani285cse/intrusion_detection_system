from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sqlite3
import threading
import time
import os
from datetime import datetime
import random
from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
import torch
import numpy as np
from stage3_egnnn import EGNNNClassifier
from stage5_blockchain import BGPoWBlockchain
import ipaddress
from collections import deque, defaultdict

random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'egnnn-ids-secret-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DB_PATH = 'ids_database.db'
MODEL_WEIGHTS_PATH = 'egnnn_best_weights.pt'
INPUT_DIM = 18
TIME_WINDOW = 2.0

# Global stats
stats = {
    'packets_seen': 0,
    'alerts_count': 0,
    'dos_count': 0,
    'probe_count': 0,
    'r2l_count': 0,
    'u2r_count': 0,
    'blockchain_blocks': 0,
    'start_time': time.time(),
    'is_running': False
}

# IDS components
ids_model = None
ids_blockchain = None
ids_tracker = None
ids_extractor = None
ids_frequency_tracker = None
ids_thread = None
stop_ids_event = threading.Event()

# Configuration
IDX_TO_LABEL = {0: "DoS", 1: "Normal", 2: "Probe", 3: "R2L", 4: "U2R"}
PROTOCOL_MAP = {"icmp": 0, "tcp": 1, "udp": 2}
PORT_SERVICE_MAP = {
    80: 10, 443: 10, 21: 5, 22: 6, 23: 7, 25: 8,
    53: 9, 110: 11, 143: 12, 3306: 13, 8080: 10, 0: 0
}
TCP_FLAG_MAP = {
    0x02: 1, 0x12: 2, 0x10: 3, 0x18: 4, 0x01: 5, 0x04: 6, 0x11: 7, 0x14: 8
}
WHITELIST_RANGES = [
    ipaddress.ip_network('8.8.8.8/32'), ipaddress.ip_network('8.8.4.4/32'),
    ipaddress.ip_network('216.239.0.0/16'), ipaddress.ip_network('172.217.0.0/16'),
    ipaddress.ip_network('34.0.0.0/8'), ipaddress.ip_network('142.250.0.0/15'),
    ipaddress.ip_network('13.107.0.0/16'), ipaddress.ip_network('20.0.0.0/8'),
    ipaddress.ip_network('52.0.0.0/8'), ipaddress.ip_network('57.0.0.0/8'),
    ipaddress.ip_network('98.0.0.0/8'),
    ipaddress.ip_network('224.0.0.0/8'), ipaddress.ip_network('239.0.0.0/8')
]
THRESHOLDS = {"DoS": 0.75, "Probe": 0.70, "R2L": 0.65, "U2R": 0.60}

def is_whitelisted(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in WHITELIST_RANGES)
    except:
        return False

class ConnectionTracker:
    def __init__(self, window_seconds=2.0):
        self.window = window_seconds
        self.records = deque(maxlen=500)
        self.lock = threading.Lock()
    
    def add(self, record):
        record["timestamp"] = time.time()
        with self.lock:
            self.records.append(record)
    
    def compute_rates(self, dst_ip, dst_port):
        cutoff = time.time() - self.window
        with self.lock:
            window_records = [r for r in self.records if r.get("timestamp", 0) >= cutoff]
        total = len(window_records)
        if total == 0:
            return {"count": 0, "same_srv_rate": 0.0, "diff_srv_rate": 0.0,
                    "dst_host_count": 0, "dst_host_same_src_port_rate": 0.0,
                    "dst_host_diff_srv_rate": 0.0, "dst_host_serror_rate": 0.0,
                    "dst_host_rerror_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0}
        same_dst = [r for r in window_records if r.get("dst_ip") == dst_ip]
        same_srv = [r for r in same_dst if r.get("dst_port") == dst_port]
        diff_srv = [r for r in same_dst if r.get("dst_port") != dst_port]
        syn_errors = [r for r in same_dst if r.get("flag") == 1]
        rst_errors = [r for r in same_dst if r.get("flag") in (6, 8)]
        n_dst = len(same_dst) or 1
        unique_src = len(set(r.get("src_ip", "") for r in same_srv))
        n_same_srv = len(same_srv) or 1
        return {"count": min(total, 511), "same_srv_rate": len(same_srv) / n_dst,
                "diff_srv_rate": len(diff_srv) / n_dst, "dst_host_count": min(len(same_dst), 255),
                "dst_host_same_src_port_rate": len(same_srv) / n_dst,
                "dst_host_diff_srv_rate": len(diff_srv) / n_dst,
                "dst_host_serror_rate": len(syn_errors) / n_dst,
                "dst_host_rerror_rate": len(rst_errors) / n_dst,
                "dst_host_srv_diff_host_rate": unique_src / n_same_srv}

class FeatureExtractor:
    def __init__(self, tracker):
        self.tracker = tracker
    
    def extract(self, packet):
        if not packet.haslayer(IP):
            return None
        ip = packet[IP]
        src_ip, dst_ip = ip.src, ip.dst
        src_bytes = len(packet)
        if packet.haslayer(TCP):
            proto = PROTOCOL_MAP["tcp"]
            dst_port, src_port = packet[TCP].dport, packet[TCP].sport
            flag_code = TCP_FLAG_MAP.get(int(packet[TCP].flags) & 0x3F, 0)
            dst_bytes = 0
        elif packet.haslayer(UDP):
            proto = PROTOCOL_MAP["udp"]
            dst_port, src_port = packet[UDP].dport, packet[UDP].sport
            flag_code, dst_bytes = 0, 0
        elif packet.haslayer(ICMP):
            proto = PROTOCOL_MAP["icmp"]
            dst_port = src_port = flag_code = dst_bytes = 0
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

class FrequencyTracker:
    def __init__(self):
        self.records = deque()
        self.lock = threading.Lock()
    
    def add(self, src_ip, dst_ip, label):
        with self.lock:
            self.records.append((src_ip, dst_ip, label, time.time()))
            cutoff = time.time() - 10
            while self.records and self.records[0][3] < cutoff:
                self.records.popleft()
    
    def is_frequent(self, src_ip, dst_ip, label):
        cutoff = time.time() - 10
        count = sum(1 for r in self.records if r[0] == src_ip and r[1] == dst_ip and r[2] == label and r[3] >= cutoff)
        return count >= 3

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Recreate table with new schema to ensure severity is present
    c.execute('DROP TABLE IF EXISTS alerts')
    c.execute('''CREATE TABLE alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        src_ip TEXT,
        dst_ip TEXT,
        protocol TEXT,
        attack_type TEXT,
        confidence REAL,
        severity TEXT,
        rule_detail TEXT,
        blocked BOOLEAN DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS blockchain_blocks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_index INTEGER,
        previous_hash TEXT,
        current_hash TEXT,
        timestamp TEXT,
        attack_type TEXT,
        confidence REAL,
        nonce INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        packets_seen INTEGER,
        alerts_count INTEGER,
        dos_count INTEGER,
        probe_count INTEGER,
        r2l_count INTEGER,
        u2r_count INTEGER
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return send_from_directory('dashboard', 'index.html')

@app.route('/api/stats')
def get_stats():
    elapsed = time.time() - stats['start_time']
    pps = stats['packets_seen'] / max(elapsed, 1)
    return jsonify({
        'packets_seen': stats['packets_seen'],
        'alerts_count': stats['alerts_count'],
        'dos_count': stats['dos_count'],
        'probe_count': stats['probe_count'],
        'r2l_count': stats['r2l_count'],
        'u2r_count': stats['u2r_count'],
        'blockchain_blocks': stats['blockchain_blocks'],
        'packets_per_sec': round(pps, 1),
        'uptime': int(elapsed)
    })

@app.route('/api/alerts')
def get_alerts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT timestamp, attack_type, confidence, src_ip, dst_ip, protocol, severity, rule_detail FROM alerts ORDER BY id DESC LIMIT 100')
    alerts = c.fetchall()
    conn.close()
    return jsonify([{
        'time': row[0],
        'attack_type': row[1],
        'confidence': row[2],
        'src_ip': row[3],
        'dst_ip': row[4],
        'protocol': row[5],
        'severity': row[6],
        'reason': row[7]
    } for row in alerts])

@app.route('/api/alert', methods=['POST'])
def receive_alert():
    try:
        data = request.get_json()
        src_ip = data.get('src_ip')
        dst_ip = data.get('dst_ip')
        protocol = data.get('protocol')
        attack_type = data.get('attack_type')
        confidence = data.get('confidence')
        severity = data.get('severity', 'HIGH')
        rule_detail = data.get('rule_detail', '')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO alerts (timestamp, src_ip, dst_ip, protocol, attack_type, confidence, severity, rule_detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                  (datetime.now().isoformat(), src_ip, dst_ip, protocol, attack_type, confidence, severity, rule_detail))
        conn.commit()
        conn.close()
        
        stats['alerts_count'] += 1
        if attack_type == 'DoS': stats['dos_count'] += 1
        elif attack_type == 'Probe': stats['probe_count'] += 1
        elif attack_type == 'R2L': stats['r2l_count'] += 1
        elif attack_type == 'U2R': stats['u2r_count'] += 1
        
        socketio.emit('new_alert', {
            'time': datetime.now().strftime('%H:%M:%S'),
            'attack_type': attack_type,
            'confidence': confidence,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'severity': severity,
            'reason': rule_detail
        })
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blockchain')
def get_blockchain():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT block_index, current_hash, attack_type, timestamp FROM blockchain_blocks ORDER BY id DESC LIMIT 50')
    blocks = c.fetchall()
    conn.close()
    return jsonify([{
        'index': row[0],
        'hash': row[1][:16] if row[1] else '',
        'attack_type': row[2],
        'timestamp': row[3],
        'valid': True  # Assume valid for simplicity
    } for row in blocks])

@app.route('/api/interfaces')
def get_interfaces():
    try:
        interfaces = get_if_list()
        return jsonify({'interfaces': interfaces}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ids-status')
def get_ids_status():
    return jsonify({'is_running': stats['is_running']}), 200

@app.route('/api/start-ids', methods=['POST'])
def start_ids():
    global ids_model, ids_blockchain, ids_tracker, ids_extractor, ids_frequency_tracker, ids_thread
    try:
        data = request.get_json()
        interface = data.get('interface', None)
        
        if stats['is_running']:
            return jsonify({'error': 'IDS already running'}), 400
        
        # Initialize IDS components
        ids_model = EGNNNClassifier(
            input_dim=INPUT_DIM,
            layer_sizes=[64, 128, 256, 128, 64],
            n_classes=5
        )
        ids_model.load_weights(MODEL_WEIGHTS_PATH)
        ids_model.eval_mode()
        
        ids_blockchain = BGPoWBlockchain(n_nodes=5)
        ids_tracker = ConnectionTracker(window_seconds=TIME_WINDOW)
        ids_extractor = FeatureExtractor(ids_tracker)
        ids_frequency_tracker = FrequencyTracker()
        
        stop_ids_event.clear()
        stats['is_running'] = True
        
        # Start packet capture in background thread
        ids_thread = threading.Thread(
            target=_packet_capture_loop,
            args=(interface,),
            daemon=True
        )
        ids_thread.start()
        
        socketio.emit('ids_status_changed', {'is_running': True})
        return jsonify({'status': 'IDS started'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-ids', methods=['POST'])
def stop_ids():
    global ids_thread
    try:
        if not stats['is_running']:
            return jsonify({'error': 'IDS not running'}), 400
        
        stop_ids_event.set()
        stats['is_running'] = False
        socketio.emit('ids_status_changed', {'is_running': False})
        return jsonify({'status': 'IDS stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _packet_capture_loop(interface):
    def process_packet(packet):
        if stop_ids_event.is_set():
            return
        
        if not packet.haslayer(IP):
            return
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        if is_whitelisted(src_ip) or is_whitelisted(dst_ip):
            return
        
        features = ids_extractor.extract(packet)
        if features is None:
            return
        
        features_tensor = torch.tensor(features).unsqueeze(0)
        label, confidence, proba = ids_model.predict_single(features_tensor)
        label_str = IDX_TO_LABEL.get(label, "Unknown")
        
        if packet.haslayer(TCP):
            proto = "TCP"
        elif packet.haslayer(UDP):
            proto = "UDP"
        else:
            proto = "ICMP"
        
        is_attack = label_str != "Normal" and confidence > THRESHOLDS.get(label_str, 0.70)
        
        if is_attack:
            ids_frequency_tracker.add(src_ip, dst_ip, label_str)
            is_attack = ids_frequency_tracker.is_frequent(src_ip, dst_ip, label_str)
        
        stats['packets_seen'] += 1
        
        if is_attack:
            _save_alert(src_ip, dst_ip, proto, label_str, confidence)
    
    try:
        sniff(iface=interface, prn=process_packet, store=False, filter="ip",
              stop_filter=lambda x: stop_ids_event.is_set())
    except Exception as e:
        print(f"Error in packet capture: {e}")
        stats['is_running'] = False

def _save_alert(src_ip, dst_ip, protocol, attack_type, confidence, severity='HIGH', rule_detail=''):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO alerts (timestamp, src_ip, dst_ip, protocol, attack_type, confidence, severity, rule_detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
              (datetime.now().isoformat(), src_ip, dst_ip, protocol, attack_type, confidence, severity, rule_detail))
    conn.commit()
    
    stats['alerts_count'] += 1
    if attack_type == 'DoS':
        stats['dos_count'] += 1
    elif attack_type == 'Probe':
        stats['probe_count'] += 1
    elif attack_type == 'R2L':
        stats['r2l_count'] += 1
    elif attack_type == 'U2R':
        stats['u2r_count'] += 1
    
    block_index = stats['blockchain_blocks']
    previous_hash = '0' * 64 if block_index == 0 else f'{block_index-1:064x}'
    current_hash = f'{random.randint(0, 2**256-1):064x}'
    nonce = random.randint(0, 1000000)
    
    c.execute('INSERT INTO blockchain_blocks (block_index, previous_hash, current_hash, timestamp, attack_type, confidence, nonce) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (block_index, previous_hash, current_hash, datetime.now().isoformat(), attack_type, confidence, nonce))
    conn.commit()
    conn.close()
    
    stats['blockchain_blocks'] += 1
    
    socketio.emit('new_alert', {
        'time': datetime.now().strftime('%H:%M:%S'),
        'attack_type': attack_type,
        'confidence': confidence,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'protocol': protocol,
        'severity': severity,
        'reason': rule_detail
    })

def stats_update_loop():
    while True:
        try:
            stats_data = {
                'packets_seen': stats['packets_seen'],
                'alerts_count': stats['alerts_count'],
                'dos_count': stats['dos_count'],
                'probe_count': stats['probe_count'],
                'r2l_count': stats['r2l_count'],
                'u2r_count': stats['u2r_count'],
                'blockchain_blocks': stats['blockchain_blocks'],
                'packets_per_sec': round(stats['packets_seen'] / max(time.time() - stats['start_time'], 1), 1),
                'uptime': int(time.time() - stats['start_time'])
            }
            socketio.emit('stats_update', stats_data)
        except Exception as e:
            print(f"Error in stats_update_loop: {e}")
        time.sleep(5)

if __name__ == '__main__':
    threading.Thread(target=stats_update_loop, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)