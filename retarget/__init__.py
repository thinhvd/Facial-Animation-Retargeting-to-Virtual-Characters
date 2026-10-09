"""Facial retargeting to arbitrary 3D face models.

Layout
------
formats/   readers (PMX) and writers (glTF) for the canonical mesh representation
canonical  the shared expression space (ARKit-52 + head pose + gaze) and payload schema
naming     name tables mapping MMD / VRM morph and bone names to canonical names
convert    CLI: model file -> canonical glTF + adapter
capture/   webcam -> canonical payload (MediaPipe Face Landmarker)
server     static files + Socket.IO relay (replaces NodeServer for the new pipeline)
"""
