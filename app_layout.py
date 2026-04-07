import sys
import os
import shutil
import cv2
import subprocess
import zipfile
import io
import json
import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import DBSCAN

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QSplitter,
                             QListWidget, QLabel, QFrame, QDialog, QLineEdit,
                             QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QVector3D

import pyqtgraph.opengl as gl


# --- BACKGROUND WORKERS ---
class ModelGenerationWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, input_path, safe_out_file, base_dir, swift_exe):
        super().__init__()
        self.input_path = input_path
        self.safe_out_file = safe_out_file
        self.base_dir = base_dir
        self.swift_exe = swift_exe

    def run(self):
        target_dir = self.input_path

        if os.path.isfile(self.input_path):
            self.log_signal.emit("Slicing video into frames...")
            target_dir = os.path.join(self.base_dir, "Input_Frames_Workspace")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.log_signal.emit("CRITICAL ERROR: Failed to open video.")
                self.finished_signal.emit(False, "")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(fps / 3))
            frame_count, saved_count = 0, 0

            while True:
                success, frame = cap.read()
                if not success: break
                if frame_count % frame_skip == 0:
                    cv2.imwrite(os.path.join(target_dir, f"frame_{saved_count:04d}.jpg"), frame)
                    saved_count += 1
                    if saved_count % 10 == 0:
                        self.log_signal.emit(f"Extracted {saved_count} frames...")
                frame_count += 1
            cap.release()
            self.log_signal.emit(f"Extracted {saved_count} frames. Handing off to Neural Engine...")

        try:
            process = subprocess.Popen(
                [self.swift_exe, target_dir, self.safe_out_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            for line in process.stdout:
                clean_line = line.strip()
                if not clean_line: continue
                if "Progress:" in clean_line:
                    self.log_signal.emit(f"Rendering {clean_line}")
                else:
                    self.log_signal.emit(clean_line)
            process.wait()

            if process.returncode == 0:
                self.log_signal.emit("\nSUCCESS! 3D Model Generated.")
                self.finished_signal.emit(True, self.safe_out_file)
            else:
                self.log_signal.emit("\nFAILED. Engine crashed.")
                self.finished_signal.emit(False, "")

        except Exception as e:
            self.log_signal.emit(f"FATAL Engine Error: {e}")
            self.finished_signal.emit(False, "")


class TextureBakingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, selected_model_path, base_dir):
        super().__init__()
        self.selected_model_path = selected_model_path
        self.base_dir = base_dir

    def run(self):
        obj_path = self.selected_model_path.replace(".usdz", ".obj")
        ply_path = self.selected_model_path.replace(".usdz", ".ply")

        if os.path.exists(ply_path):
            os.remove(ply_path)

        self.log_signal.emit(f"Extracting geometry from {os.path.basename(self.selected_model_path)}...")
        converter_exe = os.path.join(self.base_dir, "FormatConverter")
        subprocess.run([converter_exe, self.selected_model_path, obj_path])

        if not os.path.exists(obj_path):
            self.log_signal.emit("ERROR: Swift format conversion failed.")
            self.finished_signal.emit(False, "")
            return

        self.log_signal.emit("Adding textures to 3D geometry...")
        try:
            t_scene = trimesh.load(obj_path, process=False)
            t_mesh = t_scene.dump(concatenate=True) if isinstance(t_scene, trimesh.Scene) else t_scene
            texture_img = None

            with zipfile.ZipFile(self.selected_model_path, 'r') as archive:
                image_files = [f for f in archive.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                color_file = next(
                    (f for f in image_files if any(x in f.lower() for x in ['color', 'diffuse', 'albedo'])), None)
                if not color_file and image_files:
                    color_file = max(image_files, key=lambda f: archive.getinfo(f).file_size)

                if color_file:
                    self.log_signal.emit(f"Applying Color Map: {color_file}")
                    img_data = archive.read(color_file)
                    texture_img = Image.open(io.BytesIO(img_data))

            if texture_img and hasattr(t_mesh.visual, 'uv'):
                self.log_signal.emit("Applying ripped texture to UV map...")
                t_mesh.visual = trimesh.visual.TextureVisuals(uv=t_mesh.visual.uv, image=texture_img)
                t_mesh.visual = t_mesh.visual.to_color()
            else:
                self.log_signal.emit("WARNING: Could not find texture or UV map. Saving raw geometry.")

            t_mesh.export(ply_path)
            self.log_signal.emit("Color baking complete.")
            self.finished_signal.emit(True, ply_path)

        except Exception as e:
            self.log_signal.emit(f"FATAL Color Baking Error: {str(e)}")
            self.finished_signal.emit(False, "")


# --- NEW: SENSOR CALIBRATION WORKER ---
class SensorCalibrationWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, dict)

    def __init__(self, ply_path):
        super().__init__()
        self.ply_path = ply_path

    def run(self):
        self.log_signal.emit("\n--- STARTING SENSOR CALIBRATION ---")
        try:
            # 1. Load the baked mesh
            t_scene = trimesh.load(self.ply_path, process=False)
            mesh = t_scene.dump(concatenate=True) if isinstance(t_scene, trimesh.Scene) else t_scene

            # 2. Apply the exact same scale/rotation used in the Viewport so coordinates match perfectly
            mesh.apply_scale(1000.0)
            rot_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            mesh.apply_transform(rot_matrix)

            if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
                self.log_signal.emit("ERROR: Mesh has no color data. Cannot detect markers.")
                self.finished_signal.emit(False, {})
                return

            vertices = mesh.vertices
            # Trimesh colors are 0-255 uint8 arrays
            colors_8bit = mesh.visual.vertex_colors[:, :3]

            # 3. OpenCV HSV Conversion
            colors_reshaped = colors_8bit.reshape(-1, 1, 3)
            colors_hsv = cv2.cvtColor(colors_reshaped, cv2.COLOR_RGB2HSV)

            self.log_signal.emit("Hunting for physical markers...")

            # ~~~ ANCHOR POINT THRESHOLDS ~~~
            lower_bound = np.array([140, 15, 40], dtype=np.uint8)
            upper_bound = np.array([179, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(colors_hsv, lower_bound, upper_bound).flatten() > 0
            target_vertices = vertices[mask]

            if len(target_vertices) == 0:
                self.log_signal.emit("ERROR: Could not find any marker pixels matching the HSV threshold.")
                self.finished_signal.emit(False, {})
                return

            self.log_signal.emit(f"Found {len(target_vertices)} marker pixels. Clustering...")

            # 4. DBSCAN Clustering
            clustering = DBSCAN(eps=5.0, min_samples=2).fit(target_vertices)
            labels = clustering.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            self.log_signal.emit(f"Successfully detected {n_clusters} physical sensors!")

            sensor_nodes = {}
            for i in range(n_clusters):
                cluster_points = target_vertices[labels == i]
                centroid = np.mean(cluster_points, axis=0)
                sensor_id = f"sensor_{i + 1}"
                sensor_nodes[sensor_id] = centroid.tolist()
                self.log_signal.emit(
                    f"  -> {sensor_id} anchored at X:{centroid[0]:.1f}, Y:{centroid[1]:.1f}, Z:{centroid[2]:.1f}")

            # 5. Save the Permanent Map
            with open("sensor_map.json", "w") as f:
                json.dump(sensor_nodes, f, indent=4)

            self.log_signal.emit("Calibration saved to 'sensor_map.json'.")
            self.finished_signal.emit(True, sensor_nodes)

        except Exception as e:
            self.log_signal.emit(f"FATAL Calibration Error: {e}")
            self.finished_signal.emit(False, {})


# --- UI DIALOGS ---
class SetupModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Digital Twin")
        self.resize(500, 150)
        layout = QVBoxLayout(self)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select Folder or Video...")
        btn_folder = QPushButton("Folder")
        btn_video = QPushButton("Video")
        btn_folder.clicked.connect(self.browse_folder)
        btn_video.clicked.connect(self.browse_video)

        row1 = QHBoxLayout()
        row1.addWidget(self.input_edit)
        row1.addWidget(btn_folder)
        row1.addWidget(btn_video)
        layout.addLayout(row1)

        self.output_edit = QLineEdit("new_scan.usdz")
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Output Filename:"))
        row2.addWidget(self.output_edit)
        layout.addLayout(row2)

        self.btn_go = QPushButton("GENERATE MODEL")
        self.btn_go.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold; padding: 10px;")
        self.btn_go.clicked.connect(self.accept)
        layout.addWidget(self.btn_go)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder: self.input_edit.setText(folder)

    def browse_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if file: self.input_edit.setText(file)

    def get_data(self):
        return self.input_edit.text(), self.output_edit.text()


# --- MAIN APP ---
class TactileStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tactile Sensor Studio")
        self.resize(1200, 800)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.swift_executable = os.path.join(self.base_dir, "MeshGenerator")
        self.workspace_dir = os.path.join(self.base_dir, "3D_Export_Workspace")
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

        self.setup_ui()
        self.refresh_sidebar()

        self.log("System initialized. Native PyQt6 Engine Active.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- TOP TOOLBAR ---
        toolbar_layout = QHBoxLayout()
        self.btn_new = QPushButton("New Model")
        self.btn_new.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_record = QPushButton("▶ Record Event")
        self.btn_replay = QPushButton("↺ Replay Event")

        toolbar_layout.addWidget(self.btn_new)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.btn_record)
        toolbar_layout.addWidget(self.btn_replay)
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(v_splitter)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        v_splitter.addWidget(h_splitter)

        # --- SIDEBAR ---
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        sidebar_label = QLabel("WORKSPACE")
        sidebar_label.setStyleSheet("font-weight: bold; color: #888;")
        sidebar_layout.addWidget(sidebar_label)

        self.sidebar = QListWidget()
        self.sidebar.setStyleSheet("background-color: #2b2b2b; color: white; border-radius: 5px;")
        self.sidebar.itemSelectionChanged.connect(self.on_sidebar_select)
        sidebar_layout.addWidget(self.sidebar)

        # --- SENSOR CALIBRATION ---
        sidebar_layout.addWidget(QLabel(" "))  # Spacer
        calib_label = QLabel("3. SENSOR CALIBRATION")
        calib_label.setStyleSheet("font-weight: bold; color: #888;")
        sidebar_layout.addWidget(calib_label)

        self.btn_calibrate = QPushButton("CALIBRATE SENSORS")
        self.btn_calibrate.setStyleSheet(
            "background-color: #FF9500; color: white; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_calibrate.hide()
        self.btn_calibrate.clicked.connect(self.trigger_calibration)
        sidebar_layout.addWidget(self.btn_calibrate)

        h_splitter.addWidget(sidebar_widget)

        # --- VIEWPORT (PyQtGraph OpenGL) ---
        viewport_widget = QWidget()
        viewport_layout = QVBoxLayout(viewport_widget)
        viewport_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_viewer = gl.GLViewWidget()
        self.gl_viewer.setBackgroundColor('#1e1e1e')
        self.gl_viewer.opts['distance'] = 2.0
        viewport_layout.addWidget(self.gl_viewer)

        self.btn_render = QPushButton("RENDER DIGITAL TWIN")
        self.btn_render.setStyleSheet(
            "background-color: #007AFF; color: white; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_render.hide()
        self.btn_render.clicked.connect(self.trigger_render)
        viewport_layout.addWidget(self.btn_render)

        h_splitter.addWidget(viewport_widget)
        h_splitter.setSizes([250, 850])

        # --- TERMINAL ---
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet(
            "background-color: #111; color: #0f0; font-family: 'Menlo', monospace; border-radius: 5px; padding: 5px;")
        v_splitter.addWidget(self.terminal)
        v_splitter.setSizes([600, 200])

        self.btn_new.clicked.connect(self.trigger_new_model)

    def log(self, text):
        print(text)
        cursor = self.terminal.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text + "\n")
        self.terminal.setTextCursor(cursor)
        self.terminal.ensureCursorVisible()

    def refresh_sidebar(self):
        self.sidebar.clear()
        for f in os.listdir(self.workspace_dir):
            if f.endswith(".usdz"):
                self.sidebar.addItem(f)

    def on_sidebar_select(self):
        if self.sidebar.selectedItems():
            self.btn_render.show()
            self.btn_calibrate.show()
        else:
            self.btn_render.hide()
            self.btn_calibrate.hide()

    def trigger_new_model(self):
        dialog = SetupModelDialog(self)
        if dialog.exec():
            input_path, output_name = dialog.get_data()
            if not input_path: return
            if not output_name.endswith(".usdz"): output_name += ".usdz"

            safe_out_file = os.path.join(self.workspace_dir, output_name)

            self.btn_new.setEnabled(False)
            self.log("\n--- STARTING GENERATION PIPELINE ---")

            self.gen_worker = ModelGenerationWorker(input_path, safe_out_file, self.base_dir, self.swift_executable)
            self.gen_worker.log_signal.connect(self.log)
            self.gen_worker.finished_signal.connect(self.on_generation_finished)
            self.gen_worker.start()

    def on_generation_finished(self, success, output_file):
        self.btn_new.setEnabled(True)
        if success:
            self.refresh_sidebar()
            items = self.sidebar.findItems(os.path.basename(output_file), Qt.MatchFlag.MatchExactly)
            if items: self.sidebar.setCurrentItem(items[0])

    def trigger_render(self):
        selected = self.sidebar.currentItem()
        if not selected: return

        usdz_path = os.path.join(self.workspace_dir, selected.text())
        self.btn_render.setEnabled(False)
        self.btn_render.setText("EXTRACTING & BAKING...")
        self.log("\n--- STARTING RENDER PIPELINE ---")

        self.bake_worker = TextureBakingWorker(usdz_path, self.base_dir)
        self.bake_worker.log_signal.connect(self.log)
        self.bake_worker.finished_signal.connect(self.on_baking_finished)
        self.bake_worker.start()

    def on_baking_finished(self, success, ply_path):
        self.btn_render.setEnabled(True)
        self.btn_render.setText("RENDER DIGITAL TWIN")

        if not success: return

        self.log("Loading geometry into PyQtGraph Engine...")
        try:
            mesh = trimesh.load(ply_path, process=False)
            t_mesh = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh

            t_mesh.apply_scale(1000.0)
            rot_matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            t_mesh.apply_transform(rot_matrix)

            # SAVE VERTS AND FACES FOR LATER
            self.mesh_verts = t_mesh.vertices

            # Apple USDZ files use a reversed triangle winding order. For OpenGL, we reverse the columns of the
            # array to flip it.
            self.mesh_faces = t_mesh.faces[:, ::-1]

            if hasattr(t_mesh.visual, 'vertex_colors') and t_mesh.visual.vertex_colors is not None:
                colors = t_mesh.visual.vertex_colors / 255.0
            else:
                self.log("No colors found. Defaulting to solid grey.")
                colors = np.ones((len(self.mesh_verts), 4)) * 0.6
                colors[:, 3] = 1.0

            self.gl_viewer.clear()

            grid = gl.GLGridItem()
            grid.scale(100, 100, 100)
            self.gl_viewer.addItem(grid)

            # SAVE MESH ITEM REFERENCE
            self.mesh_item = gl.GLMeshItem(vertexes=self.mesh_verts, faces=self.mesh_faces, vertexColors=colors,
                                           smooth=True)
            self.gl_viewer.addItem(self.mesh_item)

            bounds = t_mesh.bounds
            center = t_mesh.centroid
            size = np.linalg.norm(bounds[1] - bounds[0])

            self.gl_viewer.opts['center'] = QVector3D(center[0], center[1], center[2])
            self.gl_viewer.opts['distance'] = size * 1.5

            self.log("Render complete! You can click and drag to rotate the model.")

        except Exception as e:
            self.log(f"FATAL Rendering Error: {e}")

    # --- TRIGGER CALIBRATION ---
    def trigger_calibration(self):
        selected = self.sidebar.currentItem()
        if not selected: return

        ply_path = os.path.join(self.workspace_dir, selected.text()).replace(".usdz", ".ply")

        if not os.path.exists(ply_path):
            self.log("ERROR: You must 'Render Digital Twin' first to extract the 3D geometry.")
            return

        if not hasattr(self, 'mesh_item'):
            self.log("ERROR: Please Render the model in the viewer before calibrating.")
            return

        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText("CALIBRATING...")

        self.calib_worker = SensorCalibrationWorker(ply_path)
        self.calib_worker.log_signal.connect(self.log)
        self.calib_worker.finished_signal.connect(self.on_calibration_finished)
        self.calib_worker.start()

    def on_calibration_finished(self, success, centroids):
        self.btn_calibrate.setEnabled(True)
        self.btn_calibrate.setText("CALIBRATE SENSORS")

        if not success: return

        self.log("Stripping texture colors... generating Digital Twin.")

        # 1. Delete the old flat-colored mesh
        if hasattr(self, 'mesh_item'):
            try:
                self.gl_viewer.removeItem(self.mesh_item)
            except ValueError:
                pass

        # Spawn a new mesh specifically configured for CAD-style rendering
        self.mesh_item = gl.GLMeshItem(
            vertexes=self.mesh_verts,
            faces=self.mesh_faces,
            color=(0.5, 0.5, 0.5, 1.0),  # Base Matte Grey
            shader='shaded',  # Enable OpenGL directional lighting
            smooth=False,  # Turn off smoothing
            drawEdges=True,  # Explicitly draw the wireframe
            edgeColor=(0.1, 0.1, 0.1, 1.0)  # Dark lines for triangle edges
        )
        self.gl_viewer.addItem(self.mesh_item)

        # 2. Draw the sensor nodes
        pos = np.array(list(centroids.values()))
        scatter_colors = np.zeros((len(pos), 4))
        scatter_colors[:, 0] = 1.0  # Red Channel
        scatter_colors[:, 3] = 1.0  # Alpha Channel

        # Safely check if the scatter item is actually in the viewer before deleting it
        if hasattr(self, 'scatter_item'):
            try:
                self.gl_viewer.removeItem(self.scatter_item)
            except ValueError:
                pass

        self.scatter_item = gl.GLScatterPlotItem(pos=pos, color=scatter_colors, size=15.0, pxMode=True)
        self.gl_viewer.addItem(self.scatter_item)

        self.log("SUCCESS: Digital Twin active. Sensors are calibrated and mapped.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TactileStudioApp()
    window.show()
    sys.exit(app.exec())