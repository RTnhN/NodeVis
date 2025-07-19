from __future__ import annotations

import os
from pathlib import Path

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


def _update_frame(idx: int) -> None:
    global frames_list, sensor_assemblies, vtk_render_window, text_actor, followers
    for i, assembly in enumerate(sensor_assemblies):
        frames = frames_list[i]
        if idx < 0 or idx >= len(frames):
            continue
        rotate_data = frames[idx]
        rot_mat = R.from_quat(rotate_data, scalar_first=True).as_matrix()
        rot_mat4x4 = np.pad(rot_mat, ((0, 1), (0, 1)), "constant")
        rot_mat4x4[3, 3] = 1
        rot_mat4x4[:3, 3] = [i * offset_spacing, 0, 0]  # Offset X
        temp_matrix = vtkMatrix4x4Customized(rot_mat4x4)
        if hasattr(assembly, "SetUserMatrix"):
            assembly.SetUserMatrix(temp_matrix)
    text_actor.SetInput(f"Frame: {idx}")
    vtk_render_window.Render()
    camera = vtk_renderer.GetActiveCamera()
    for follower in followers:
        follower.SetCamera(camera)


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
    board_file_name: Path, nframes: int, nsensors: int, sensor_names: list[str]
):
    global text_actor, vtk_render_window, vtk_renderer, sensor_assemblies, followers
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
    text_actor.SetInput("Frame: 0")
    text_actor.GetTextProperty().SetFontSize(36)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.SetDisplayPosition(40, 700)
    overlay_renderer.AddViewProp(text_actor)

    vtk_render_window_interactor.Initialize()
    vtk_render_window.Render()

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

    vtk_render_window.Render()
    camera = vtk_renderer.GetActiveCamera()
    for follower in followers:
        follower.SetCamera(camera)

    _update_frame(0)
    vtk_render_window_interactor.AddObserver("KeyPressEvent", zoom_to_node)
    vtk_render_window_interactor.Start()


def zoom_to_node(obj, event):
    # Check if Ctrl is held
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


def main():
    global frames_list
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize CSV quaternions with VTK.",
    )
    parser.add_argument(
        "SageMotion_data_file",
        help="SageMotion Data File (.csv or .xlsx)",
    )
    args = parser.parse_args()
    if args.SageMotion_data_file.endswith(".csv"):
        sensor_data = pd.read_csv(args.SageMotion_data_file)
    elif args.SageMotion_data_file.endswith(".xlsx"):
        sensor_data = pd.read_excel(args.SageMotion_data_file)
    else:
        raise ValueError("Invalid file type. Must be .csv or .xlsx")
    # Find all sensors
    sensor_names = [
        c.split("Quat1_")[1] for c in sensor_data.columns if c.startswith("Quat1_")
    ]
    nsensors = len(sensor_names)
    frames_list = [
        sensor_data[
            [f"Quat1_{name}", f"Quat2_{name}", f"Quat3_{name}", f"Quat4_{name}"]
        ].to_numpy()
        for name in sensor_names
    ]
    nframes = len(frames_list[0])
    _init_3D_scene("Node.glb", nframes, nsensors, sensor_names)


if __name__ == "__main__":
    main()
