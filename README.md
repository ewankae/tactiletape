# tactiletape
## Requirements
1. Apple Silicon Mac  
2. A copy of Python  
3. MiniConda (see download instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install/overview))  
## Download Instructions
1. Download the code  
a. Click the green "<> Code" button near the top right.  
b. Select "download ZIP"  
c. Locate the ZIP, double-click it to extract it, and move the unzipped folder to your desktop.  

2. Navigate in the terminal
```
cd ~/Desktop/tactiletape
```
(Note: you can forgo downloading the ZIP and navigating to the extracted folder by cloning into the git repo, if you know how to do so)

3. Create a Conda environment
```
conda create -n tactile_mac python=3.11 -y
conda activate tactile_mac
```

4. Install Python dependencies
```
pip install --upgrade pip
pip install PyQt6 pyqtgraph PyOpenGL
pip install open3d opencv-python trimesh Pillow scikit-learn
```

5. Grant file executable permissions  


Note that this is only necessary when pulling the repo for the first time.
```
chmod +x MeshGenerator FormatConverter
```

## Usage
To run the app, open the terminal and activate the conda environment before launching the Python file:
```
conda activate tactile_mac
python app_layout.py
```
Not all buttons are functioning yet, but the *New Model*, *Render Digital Twin*, and *Calibrate Sensor* buttons should be working (you can only access the latter two buttons when a model is selected in the left toolbar).
