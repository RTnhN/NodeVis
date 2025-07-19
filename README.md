# SageMotion NodeVis Node Orientation Visualizer

A cross-platform Python tool for visualizing SageMotion (or other IMU) quaternion CSV/XLSX data with synchronized 3D model animation using VTK.

---

## Features

* Visualize up to 8 nodes at once in a single window, each using the same 3D model
* Real-time quaternion-driven animation with a frame slider
* Camera controls via mouse and keyboard
* 3D orientation axis and sensor index labels
* Camera shortcuts: Ctrl+1 through Ctrl+8 jump to each node

---


## Installation (with [uv](https://github.com/astral-sh/uv))

1. **Install [uv](https://github.com/astral-sh/uv):**

   ```sh
   pip install uv
   ```

2. **Install required Python packages:**

   ```sh
   uv sync
   ```

---

## Usage

1. Prepare your IMU data in CSV or XLSX format with columns for each sensor:

   * Example column names: `Quat1_SENSOR`, `Quat2_SENSOR`, `Quat3_SENSOR`, `Quat4_SENSOR` (one set per sensor).
   * The script auto-detects available sensors by scanning these columns.

2. Run the script:

   ```sh
   uv run python NodeVis.py your_data.csv
   # or
   uv run python NodeVis.py your_data.xlsx
   ```

3. **Controls:**

   * **Mouse**: Left drag = rotate, Scroll = zoom, Mouse wheel hold and drag = pan
   * **Slider** (bottom): Scrub through frames interactively
   * **Ctrl+1 ... Ctrl+8**: Instantly zoom camera to any node
   * **3D orientation axis**: See axes in the lower left
   * **Frame number**: Displayed at the top left
   * **Node labels**: Each model has a yellow label above it

---

## Notes

* Large datasets (thousands of frames) may use significant RAM/GPU.

---

## License

MIT License


