# OpenVtuber
Real-time facial animation transfer system using Python, TensorFlow Lite, and Three.js.

## Architecture
```text
Python (Face Tracking) -> Socket.IO -> Node.js (Relay) -> Browser (Three.js MMD)
```

## Setup Steps
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Node.js dependencies:
   ```bash
   cd NodeServer
   yarn install
   ```

## Running the Application
1. Start the Node.js server:
   ```bash
   cd NodeServer
   yarn start
   ```
2. Start the Python face tracker:
   - For webcam: `python vtuber_link_start.py 0`
   - For video: `python vtuber_link_start.py path/to/video.mp4`
3. Open your browser to `http://127.0.0.1:6789/` AFTER the Python script has successfully started and shows the face mesh.

## Troubleshooting
- **NumPy Warning/Crash**: Ensure you use the exact pinned requirements; NumPy 1.24+ removes deprecated types like `np.int`.
- **Browser Initializing**: The browser frontend expects the Python socket to stream data. It is best to open the browser only after Python outputs bounding boxes or connections.
