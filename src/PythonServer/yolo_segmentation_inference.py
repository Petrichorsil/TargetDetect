import cv2
import socket
import threading
import time
import torch
import sys
from ultralytics import YOLO

# IMPORTANT: Make sure this path points to your trained YOLOv8 SEGMENTATION model.
# For example: '/path/to/your/yolov8n-seg.pt'
MODEL_PATH = "./12_5.pt"
#MODEL_PATH = "./best.pt" 
HOST = '127.0.0.1'
PORT = 65432

class YOLOv8InstanceSegmentationServer:
    # In the __init__ method, modifications were made
    def __init__(self):
        self.model = None
        self._load_model()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Attempt to bind the port, retrying on failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.server_socket.bind((HOST, PORT))
                self.server_socket.listen(1)
                print(f"Server listening on {HOST}:{PORT}", file=sys.stderr)
                break
            except OSError as e:
                if attempt == max_retries - 1:
                    print(f"Failed to bind after {max_retries} attempts: {e}", file=sys.stderr)
                    sys.exit(1)
                print(f"Port in use, retrying... ({attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(1)
        
    def _load_model(self):
        """Loads the YOLOv8 model and forces it to use CUDA"""
        print(f"Active CUDA Device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device not found. Please check GPU drivers.")
            
            self.model = YOLO(MODEL_PATH).to('cuda')
            # Warm-up inference
            dummy_image = torch.zeros((640, 640, 3), dtype=torch.uint8).numpy()
            _ = self.model(dummy_image, device='cuda:0', verbose=False)
            torch.cuda.synchronize()
            
            print(f"Model loaded successfully on CUDA", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"Error loading model: {str(e)}", file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)
    
    def process_image(self, image_path):
        """Processes a single image for instance segmentation"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "ERROR: Could not read image\n"
                
            start_time = time.time()
            results = self.model(image, device='cuda:0', verbose=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"Detection time: {elapsed:.4f} seconds", file=sys.stderr)
            sys.stderr.flush()
            
            # --- MODIFIED LOGIC FOR INSTANCE SEGMENTATION ---
            detections = []
            if results[0].masks is not None:
                # Iterate over each detected instance
                for i in range(len(results[0].masks)):
                    box = results[0].boxes[i]
                    if box.conf > 0.5:
                        # Get the class ID
                        cls = int(box.cls)
                        
                        # Get the segmentation mask polygon points
                        mask_polygon = results[0].masks.xy[i]
                        polygon_points = mask_polygon.flatten().astype(int)
                        
                        # Format as: class_id,x1,y1,x2,y2,...
                        points_str = ",".join(map(str, polygon_points))
                        detections.append(f"{cls},{points_str}")
                        
            return "\n".join(detections) + "\nEND\n"
        except Exception as e:
            print(f"Error during inference: {str(e)}", file=sys.stderr)
            sys.stderr.flush()
            return "ERROR: Inference failed\nEND\n"

    def handle_client(self, conn):
        """Handles a client connection"""
        try:
            while True:
                # Receive the image path
                data = conn.recv(1024).decode('utf-8').strip()
                if not data:
                    break
                
                if data == "exit":
                    break
                
                print(f"Received image path: {data}", file=sys.stderr)
                result = self.process_image(data)
                conn.sendall(result.encode('utf-8'))
        finally:
            conn.close()

    def run(self):
        print("YOLOv8 instance segmentation server is ready", file=sys.stderr)
        try:
            while True:
                try:
                    conn, addr = self.server_socket.accept()
                    print(f"Connected by {addr}", file=sys.stderr)
                    client_thread = threading.Thread(target=self.handle_client, args=(conn,))
                    client_thread.daemon = True
                    client_thread.start()
                except OSError as e:
                    if self.server_socket._closed:
                        break
                    print(f"Accept error: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nServer shutting down...", file=sys.stderr)
        finally:
            self.server_socket.close()
            print("Server socket closed", file=sys.stderr)
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    server = YOLOv8InstanceSegmentationServer()
    server.run()
