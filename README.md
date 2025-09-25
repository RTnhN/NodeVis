# SageMotion NodeVis Node Orientation Visualizer

A cross-platform Python tool for visualizing SageMotion (or other IMU) quaternion CSV/XLSX/STO data with synchronized 3D model animation using VTK.

![vid](https://github.com/user-attachments/assets/8751fccd-91cf-42fd-b72f-9dd34a2415aa)
---


## Features

* Visualize up to 8 nodes at once in a single window, each using the same 3D model
* Real-time quaternion-driven animation with a frame slider
* Camera controls via mouse and keyboard
* 3D orientation axis and sensor index labels
* Camera shortcuts: Ctrl+1 through Ctrl+8 jump to each node
* Red circle is the spin center, which is the center of the camera's view

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

### Run directly with data file

1. Prepare your IMU data in CSV, XLSX, or STO format with columns for each sensor:

   * Example column names: `Quat1_SENSOR`, `Quat2_SENSOR`, `Quat3_SENSOR`, `Quat4_SENSOR` (one set per sensor). Quaternions are in the scalar-first order (w, x, y, z).
   * The script auto-detects available sensors by scanning these columns.
   * For `.sto` files, every column (except the `time` column) should contain a comma- or whitespace-separated quaternion value such as `0.4005,0.6229,0.5504,0.3856`.

2. Run the script:

   ```sh
   uv run python NodeVis.py your_data.csv
   # or
   uv run python NodeVis.py your_data.sto
   # or
   uv run python NodeVis.py your_data.xlsx
   ```

### Install / uninstall Windows context menu entry

You can add **"Open in NodeViz"** to the Windows right-click menu for files.
This will call the script directly when you right-click a supported file type.

* To **install** the context menu entry:

  ```sh
  uv run python NodeVis.py --install-context-menu
  ```

* To **uninstall** the context menu entry:

  ```sh
  uv run python NodeVis.py --uninstall-context-menu
  ```

After installation, right-clicking a file in Explorer will show **Open in NodeViz** under the `NodeViz` context menu group.

---

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

