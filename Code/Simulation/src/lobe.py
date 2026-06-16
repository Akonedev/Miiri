import zmq
import os
import time
import signal
import sys

LOBE_ID = os.environ.get('LOBE_ID', 'GENERIC_LOBE')
PORT = os.environ.get('PORT', '26400')
TYPE = os.environ.get('TYPE', 'SUB')

def signal_handler(sig, frame):
    print(f"[{LOBE_ID}] Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_node():
    context = zmq.Context()
    
    if TYPE == 'PUB':
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://*:{PORT}")
        print(f"[{LOBE_ID}] Lobe active (Publisher) on port {PORT}")
    elif TYPE == 'SUB':
        socket = context.socket(zmq.SUB)
        # In a real setup, we would connect to the supervisor
        # For now, we bind to allow external monitoring
        socket.bind(f"tcp://*:{PORT}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[{LOBE_ID}] Lobe active (Subscriber) on port {PORT}")
    elif TYPE == 'REP':
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{PORT}")
        print(f"[{LOBE_ID}] Lobe active (Replier) on port {PORT}")
    
    print(f"[{LOBE_ID}] Miiri-256 Neural Link Initialized.")
    
    while True:
        try:
            if TYPE == 'REP':
                message = socket.recv_string()
                print(f"[{LOBE_ID}] Received request: {message}")
                socket.send_string(f"ACK: {LOBE_ID} processed {message}")
            else:
                time.sleep(10)
                if TYPE == 'PUB':
                    socket.send_string(f"HEARTBEAT from {LOBE_ID}")
        except zmq.ZMQError as e:
            print(f"[{LOBE_ID}] ZMQ Error: {e}")
            break

if __name__ == "__main__":
    run_node()
