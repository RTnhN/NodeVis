from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
from time import monotonic

CURRENT_DIR = Path(__file__).parent
CLIENT_ROOT = CURRENT_DIR / "SageMotion_Client"
if CLIENT_ROOT.exists():
    sys.path.insert(0, str(CLIENT_ROOT))

try:
    from sagemotion_client.DataStreamClient import DataStreamClient
except ModuleNotFoundError:
    from DataStreamClient import DataStreamClient

import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D, vtkSliderWidget
from vtkmodules.vtkRenderingCore import vtkAssembly, vtkFollower

# Globals
vtk_render_window: vtk.vtkRenderWindow | None = None
vtk_renderer: vtk.vtkRenderer | None = None
text_actor: vtk.vtkTextActor | None = None
frames_list: list[np.ndarray] = []
sensor_assemblies: list[vtkAssembly] = []
offset_spacing: float = 0.2  # space between each IMU model
followers: list[vtkFollower] = []
spin_center_actor: vtk.vtkActor | None = None  # **SPIN CENTER**
slider_widget: vtk.vtkSliderWidget | None = None
STREAM_SENSOR_REQUEST_COUNT = 8
STREAM_UPDATE_INTERVAL_SECONDS = 1 / 20


class vtkMatrix4x4Customized(vtk.vtkMatrix4x4):
    def __init__(self, mat: np.ndarray | None = None) -> None:
        super().__init__()
        if mat is not None and isinstance(mat, np.ndarray):
            self.SetElements(mat)

    def SetElements(self, mat: np.ndarray) -> None:
        if mat.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        for i in range(4):
            for j in range(4):
                self.SetElement(i, j, mat[i, j])


def update_spin_center() -> None:
    global spin_center_actor, vtk_renderer
    if spin_center_actor and vtk_renderer:
        camera = vtk_renderer.GetActiveCamera()
        fp = camera.GetFocalPoint()
        spin_center_actor.SetPosition(*fp)
        vtk_render_window.Render()


def _apply_sensor_rotations(
    quaternions: Sequence[np.ndarray | Sequence[float] | None],
) -> None:
    global sensor_assemblies
    for i, assembly in enumerate(sensor_assemblies):
        if i >= len(quaternions):
            continue
        quat_value = quaternions[i]
        if quat_value is None:
            continue
        quat = np.asarray(quat_value, dtype=float)
        if quat.shape != (4,):
            continue
        rot_mat = R.from_quat(quat, scalar_first=True).as_matrix()
        rot_mat4x4 = np.pad(rot_mat, ((0, 1), (0, 1)), "constant")
        rot_mat4x4[3, 3] = 1
        rot_mat4x4[:3, 3] = [i * offset_spacing, 0, 0]
        temp_matrix = vtkMatrix4x4Customized(rot_mat4x4)
        if hasattr(assembly, "SetUserMatrix"):
            assembly.SetUserMatrix(temp_matrix)


def _refresh_scene() -> None:
    if vtk_render_window:
        vtk_render_window.Render()
    if vtk_renderer:
        camera = vtk_renderer.GetActiveCamera()
        for follower in followers:
            follower.SetCamera(camera)
    update_spin_center()  # **SPIN CENTER**


def _parse_quaternion_string(raw_value: object) -> np.ndarray:
    if pd.isna(raw_value):
        raise ValueError("Encountered NaN quaternion value.")
    raw_text = str(raw_value).strip()
    if not raw_text:
        raise ValueError("Encountered empty quaternion value.")
    if "," in raw_text:
        parts = [part.strip() for part in raw_text.split(",")]
    else:
        parts = raw_text.split()
    if len(parts) != 4:
        raise ValueError(
            f"Quaternion '{raw_text}' does not have four components.",
        )
    try:
        return np.array([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise ValueError(
            f"Quaternion '{raw_text}' contains non-numeric data.",
        ) from exc


def _load_csv_or_excel(file_path: Path) -> tuple[list[str], list[np.ndarray]]:
    if file_path.suffix.lower() == ".csv":
        sensor_data = pd.read_csv(file_path)
    else:
        sensor_data = pd.read_excel(file_path)

    sensor_names = [
        column.split("Quat1_")[1]
        for column in sensor_data.columns
        if column.startswith("Quat1_")
    ]
    if not sensor_names:
        raise ValueError(
            "No sensors were detected. Columns must follow the 'Quat1_NAME' pattern.",
        )
    frames = [
        sensor_data[
            [f"Quat1_{name}", f"Quat2_{name}", f"Quat3_{name}", f"Quat4_{name}"]
        ].to_numpy()
        for name in sensor_names
    ]
    return sensor_names, frames


def _load_sto(file_path: Path) -> tuple[list[str], list[np.ndarray]]:
    with file_path.open("r", encoding="utf-8") as sto_file:
        header_end_index = None
        for idx, line in enumerate(sto_file):
            if line.strip().lower() == "endheader":
                header_end_index = idx
                break
    if header_end_index is None:
        raise ValueError(f"The file '{file_path}' is missing an 'endheader' line.")

    sto_data = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=header_end_index + 1,
        engine="python",
    )
    sto_data.columns = [str(column).strip() for column in sto_data.columns]

    sensor_names: list[str] = []
    frames: list[np.ndarray] = []
    for column in sto_data.columns:
        if column.lower() == "time":
            continue
        column_series = sto_data[column]
        valid_values = [
            str(value).strip()
            for value in column_series
            if not pd.isna(value) and str(value).strip()
        ]
        if not valid_values:
            continue
        if valid_values[0].count(",") != 3 and len(valid_values[0].split()) != 4:
            continue
        try:
            quaternions = np.vstack(
                [_parse_quaternion_string(value) for value in column_series]
            )
        except ValueError as exc:
            raise ValueError(
                f"Column '{column}' in '{file_path}' does not contain valid quaternion data.",
            ) from exc
        sensor_names.append(column)
        frames.append(quaternions)

    if not frames:
        raise ValueError(
            f"No quaternion columns were found in '{file_path}'.",
        )
    return sensor_names, frames


def _load_sensor_data(file_path: Path) -> tuple[list[str], list[np.ndarray]]:
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".xlsx"}:
        return _load_csv_or_excel(file_path)
    if suffix == ".sto":
        return _load_sto(file_path)
    raise ValueError("Invalid file type. Must be .csv, .xlsx, or .sto")


def _update_frame(idx: int) -> None:
    global frames_list, text_actor
    if idx < 0 or not frames_list:
        return
    quaternions: list[np.ndarray] = []
    for frames in frames_list:
        if idx >= len(frames):
            quaternions.append(frames[-1])
        else:
            quaternions.append(frames[idx])
    _apply_sensor_rotations(quaternions)
    if text_actor:
        text_actor.SetInput(f"Frame: {idx}")
    _refresh_scene()


def _slider_callback(obj: vtk.vtkSliderWidget, event: vtk.vtkObject) -> None:
    idx = int(obj.GetRepresentation().GetValue())
    _update_frame(idx)


def _add_slider_widget(
    iren: vtk.vtkRenderWindowInteractor, nframes: int
) -> vtk.vtkSliderWidget:
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(nframes - 1)
    slider_rep.SetValue(0)
    slider_rep.SetTitleText("Frame")
    slider_rep.SetLabelFormat("%.0f")
    slider_rep.GetSliderProperty().SetColor(1, 0, 0)
    slider_rep.GetTitleProperty().SetColor(1, 1, 1)
    slider_rep.GetLabelProperty().SetColor(1, 1, 1)
    slider_rep.GetSelectedProperty().SetColor(0, 1, 0)
    slider_rep.GetTubeProperty().SetColor(1, 1, 1)
    slider_rep.GetCapProperty().SetColor(1, 1, 1)
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.04)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTubeWidth(0.008)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(0.2, 0.08)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(0.8, 0.08)

    slider_widget = vtkSliderWidget()
    slider_widget.SetInteractor(iren)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToJump()
    slider_widget.EnabledOn()
    slider_widget.AddObserver("InteractionEvent", _slider_callback)
    slider_widget.EnabledOff()
    slider_widget.EnabledOn()
    return slider_widget


def _init_3D_scene(
    board_file_name: Path,
    nframes: int,
    nsensors: int,
    sensor_names: list[str],
    *,
    enable_slider: bool = True,
    initial_text: str = "Frame: 0",
) -> vtk.vtkRenderWindowInteractor:
    global \
        text_actor, \
        vtk_render_window, \
        vtk_renderer, \
        sensor_assemblies, \
        followers, \
        spin_center_actor, \
        slider_widget
    data_root = Path(__file__).parent
    importer = vtk.vtkGLTFImporter()
    importer.SetFileName(str(Path(data_root) / board_file_name))
    importer.Update()

    vtk_renderer = importer.GetRenderer()
    vtk_render_window = importer.GetRenderWindow()
    vtk_render_window.SetNumberOfLayers(2)

    overlay_renderer = vtk.vtkRenderer()
    overlay_renderer.SetLayer(1)
    overlay_renderer.InteractiveOff()
    vtk_render_window.AddRenderer(overlay_renderer)

    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    style = vtk.vtkInteractorStyleTrackballCamera()
    vtk_render_window_interactor.SetInteractorStyle(style)

    vtk_renderer.GradientBackgroundOn()
    default_camera = vtk_renderer.GetActiveCamera()
    default_camera.SetPosition(0, -3, 0)
    default_camera.SetFocalPoint((nsensors - 1) * offset_spacing / 2, 0, 0)
    default_camera.SetViewUp(0, 0, 1)
    default_camera.SetClippingRange(0.1, 2000.0)
    vtk_renderer.SetBackground(0.2, 0.2, 0.2)
    vtk_renderer.SetBackground2(0.3, 0.3, 0.3)
    vtk_renderer.SetAmbient(1, 1, 1)
    vtk_render_window.SetSize(1000, 800)
    vtk_render_window.SetWindowName("SageMotion CSV Playback Demo")

    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(initial_text)
    text_actor.GetTextProperty().SetFontSize(36)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.SetDisplayPosition(40, 700)
    overlay_renderer.AddViewProp(text_actor)

    vtk_render_window_interactor.Initialize()
    vtk_render_window.Render()

    slider_widget = None
    if enable_slider and nframes > 0:
        slider_widget = _add_slider_widget(vtk_render_window_interactor, nframes)
        slider_widget.EnabledOff()
        slider_widget.EnabledOn()
        vtk_render_window.Render()

    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(200, 200, 200)
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(vtk_render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.25, 0.25)
    widget.SetEnabled(1)
    widget.InteractiveOff()

    # --- Group all imported props as an assembly per sensor ---
    imported_props = []
    props = vtk_renderer.GetViewProps()
    for i in range(props.GetNumberOfItems()):
        prop = props.GetItemAsObject(i)
        if isinstance(prop, vtk.vtkProp3D):
            imported_props.append(prop)

    sensor_assemblies = []
    for i in range(nsensors):
        assembly = vtkAssembly()
        for prop in imported_props:
            if i == 0:
                vtk_renderer.RemoveViewProp(prop)
                assembly.AddPart(prop)
            else:
                # Safely copy the actor
                actor = vtk.vtkActor.SafeDownCast(prop)
                if actor:
                    new_actor = vtk.vtkActor()
                    new_actor.ShallowCopy(actor)
                    assembly.AddPart(new_actor)
        vtk_renderer.AddViewProp(assembly)
        sensor_assemblies.append(assembly)

    followers = []
    for i, _ in enumerate(sensor_names):
        vector_text = vtk.vtkVectorText()
        vector_text.SetText(str(i + 1))
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(vector_text.GetOutputPort())
        follower = vtkFollower()
        follower.SetMapper(text_mapper)
        follower.SetScale(0.1, 0.1, 0.1)
        follower.SetPosition((i * offset_spacing) - 0.01, 0.01, 0.1)
        follower.GetProperty().SetColor(1, 1, 0)
        vtk_renderer.AddActor(follower)
        followers.append(follower)

    # Single world-axis reference at the global origin (not duplicated per node).
    axis_length = 0.15
    world_axes = vtk.vtkAxesActor()
    world_axes.SetTotalLength(axis_length, axis_length, axis_length)
    world_axes.SetAxisLabels(0)
    axes_offset = (-offset_spacing * 1.5, 0.0, 0.0)
    axes_transform = vtk.vtkTransform()
    axes_transform.Translate(*axes_offset)
    world_axes.SetUserTransform(axes_transform)
    vtk_renderer.AddActor(world_axes)

    # Label each axis with a letter so the global orientation is explicit.
    axis_labels = (
        ("X", (1.0, 0.2, 0.2), (axis_length + 0.02, 0.0, 0.0)),
        ("Y", (0.2, 1.0, 0.2), (0.0, axis_length + 0.02, 0.0)),
        ("Z", (0.2, 0.6, 1.0), (0.0, 0.0, axis_length + 0.02)),
    )
    for label_text, color, offset_vec in axis_labels:
        vector_text = vtk.vtkVectorText()
        vector_text.SetText(label_text)
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(vector_text.GetOutputPort())
        follower = vtkFollower()
        follower.SetMapper(text_mapper)
        follower.SetScale(0.05, 0.05, 0.05)
        follower.SetPosition(
            axes_offset[0] + offset_vec[0],
            axes_offset[1] + offset_vec[1],
            axes_offset[2] + offset_vec[2],
        )
        follower.GetProperty().SetColor(color)
        vtk_renderer.AddActor(follower)
        followers.append(follower)

    spin_center = vtk.vtkSphereSource()
    spin_center.SetRadius(0.005)
    spin_center.SetThetaResolution(16)
    spin_center.SetPhiResolution(16)
    spin_center_mapper = vtk.vtkPolyDataMapper()
    spin_center_mapper.SetInputConnection(spin_center.GetOutputPort())
    spin_center_actor = vtk.vtkActor()
    spin_center_actor.SetMapper(spin_center_mapper)
    spin_center_actor.GetProperty().SetColor(1, 0, 0)
    fp = default_camera.GetFocalPoint()
    spin_center_actor.SetPosition(*fp)
    vtk_renderer.AddActor(spin_center_actor)

    vtk_render_window.Render()
    camera = vtk_renderer.GetActiveCamera()
    for follower in followers:
        follower.SetCamera(camera)

    vtk_render_window_interactor.AddObserver("KeyPressEvent", zoom_to_node)
    vtk_render_window_interactor.AddObserver(
        "InteractionEvent", lambda o, e: update_spin_center()
    )
    return vtk_render_window_interactor


def zoom_to_node(obj, event):
    if obj.GetControlKey():
        key = obj.GetKeySym()
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(sensor_assemblies):
                pos = sensor_assemblies[idx].GetCenter()
                camera = vtk_renderer.GetActiveCamera()
                camera.SetFocalPoint(*pos)
                camera.SetPosition(pos[0], pos[1] - 0.6, pos[2] + 0.2)
                camera.SetViewUp(0, 0, 1)
                vtk_renderer.ResetCameraClippingRange()
                vtk_render_window.Render()
                update_spin_center()  # **SPIN CENTER**


def _run_viewer(data_file: Path) -> None:
    global frames_list
    if not data_file.exists():
        raise FileNotFoundError(f"Could not find data file '{data_file}'.")
    sensor_names, loaded_frames = _load_sensor_data(data_file)
    if not loaded_frames:
        raise ValueError("No sensor quaternion data was found in the provided file.")
    frame_counts = {frame.shape[0] for frame in loaded_frames}
    if len(frame_counts) != 1:
        raise ValueError("All sensors must contain the same number of frames.")
    frames_list = loaded_frames
    nsensors = len(sensor_names)
    nframes = frame_counts.pop()
    interactor = _init_3D_scene(Path("Node.glb"), nframes, nsensors, sensor_names)
    _update_frame(0)
    interactor.Start()


def _extract_quaternions_from_raw_data(raw_data: object) -> list[np.ndarray | None]:
    quat_keys = ("Quat1", "Quat2", "Quat3", "Quat4")
    if not isinstance(raw_data, Sequence) or isinstance(raw_data, (bytes, str)):
        return []
    quaternions: list[np.ndarray | None] = []
    for sensor in raw_data:
        if not isinstance(sensor, dict):
            quaternions.append(None)
            continue
        try:
            quat = np.array([float(sensor[key]) for key in quat_keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            quaternions.append(None)
            continue
        quaternions.append(quat)
    return quaternions


def _run_stream_viewer(
    stream_address: str,
    *,
    stream_port: int = 5678,
    request_log_time: bool = False,
) -> None:
    from queue import Empty, Queue
    import threading

    global frames_list
    frames_list = []

    data_queue: Queue[tuple[int, dict[str, object]]] = Queue()
    fields = ("Quat1", "Quat2", "Quat3", "Quat4")
    requested_data: list[list[object]] = [
        [sensor_idx, field]
        for sensor_idx in range(STREAM_SENSOR_REQUEST_COUNT)
        for field in fields
    ]
    requested_data.append([None, "raw_data"])

    if request_log_time:
        requested_data.append([None, "universal_time"])

    client = DataStreamClient(
        stream_address,
        queue=data_queue,
        port=stream_port,
        requested_data=requested_data,
    )
    client_thread = threading.Thread(target=client.run_forever, daemon=True)
    client_thread.start()

    raw_data: object | None = None
    try:
        while True:
            _, first_payload = data_queue.get(timeout=10)
            raw_data = (
                first_payload.get("raw_data")
                if isinstance(first_payload, dict)
                else None
            )
            quaternions = _extract_quaternions_from_raw_data(raw_data)
            if quaternions:
                break
    except Empty as exc:
        raise RuntimeError(
            "No data received from the SageMotion stream within 10 seconds.",
        ) from exc

    if not isinstance(raw_data, Sequence) or isinstance(raw_data, (bytes, str)):
        raw_data = []
    nsensors = len(raw_data) if raw_data else len(quaternions)
    if nsensors == 0:
        raise ValueError("Streaming data did not contain any sensors.")

    sensor_names = [str(i + 1) for i in range(nsensors)]
    interactor = _init_3D_scene(
        Path("Node.glb"),
        1,
        nsensors,
        sensor_names,
        enable_slider=False,
        initial_text="Streaming frame: 0",
    )

    _apply_sensor_rotations(quaternions)
    frame_counter = 1
    last_render_time = 0.0
    latest_payload: dict[str, object] | None = None
    if text_actor:
        text_actor.SetInput(f"Streaming frame: {frame_counter}")
    _refresh_scene()

    def _drain_queue(obj, event):
        nonlocal frame_counter, last_render_time
        processed = False
        while True:
            try:
                _, payload = data_queue.get_nowait()
            except Empty:
                break
            raw = payload.get("raw_data") if isinstance(payload, dict) else None
            new_quaternions = _extract_quaternions_from_raw_data(raw)
            if not new_quaternions:
                continue
            _apply_sensor_rotations(new_quaternions)
            frame_counter += 1
            processed = True
        if not processed:
            return
        now = monotonic()
        if now - last_render_time > STREAM_UPDATE_INTERVAL_SECONDS:
            last_render_time = now
            if text_actor:
                text_actor.SetInput(f"Streaming frame: {frame_counter}")
            _refresh_scene()

    interactor.AddObserver("TimerEvent", _drain_queue)
    interactor.CreateRepeatingTimer(5)
    interactor.Start()


def _context_launch(filenames, params):
    if not filenames:
        return
    print("Running NodeViz viewer with the following file:", filenames[0])
    _run_viewer(Path(filenames[0]))


def _install_context_menu() -> None:
    from context_menu import menus

    cm = menus.ContextMenu("NodeViz", type="FILES")
    cm.add_items(
        [
            menus.ContextCommand("Open in NodeViz", python=_context_launch),
        ],
    )
    cm.compile()


def _uninstall_context_menu() -> None:
    from context_menu import menus

    cm = menus.ContextMenu("NodeViz", type="FILES")
    cm.add_items(
        [
            menus.ContextCommand("Open in NodeViz", python=_context_launch),
        ]
    )
    cm.remove()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize IMU quaternion data with VTK.",
    )
    parser.add_argument(
        "SageMotion_data_file",
        help="SageMotion Data File (.csv, .xlsx, or .sto)",
        nargs="?",
    )
    parser.add_argument(
        "--install-context-menu",
        action="store_true",
        help="Install a right-click context menu entry that opens selected files in this viewer.",
    )

    parser.add_argument(
        "--uninstall-context-menu",
        action="store_true",
        help="Remove the NodeViz entry from the right-click context menu.",
    )
    parser.add_argument(
        "--stream-address",
        help="IP address of a SageMotion stream provider to visualize live data.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=5678,
        help="Port for the streaming data source (default: 5678).",
    )
    parser.add_argument(
        "--stream-log-time",
        action="store_true",
        help="Request universal_time from the stream alongside quaternions.",
    )
    args = parser.parse_args()

    if args.install_context_menu:
        _install_context_menu()
        return

    if args.uninstall_context_menu:
        _uninstall_context_menu()
        return

    if args.stream_address:
        _run_stream_viewer(
            args.stream_address,
            stream_port=args.stream_port,
            request_log_time=args.stream_log_time,
        )
        return

    if not args.SageMotion_data_file:
        parser.error(
            "You must provide a SageMotion data file or use --stream-address.",
        )
    _run_viewer(Path(args.SageMotion_data_file))


if __name__ == "__main__":
    main()
