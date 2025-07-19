from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import vtk
from scipy.spatial.transform import Rotation as R  # noqa: N817
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D, vtkSliderWidget

# Global variables
vtk_render_window: vtk.vtkRenderWindow | None = None
vtk_renderer: vtk.vtkRenderer | None = None
text_actor: vtk.vtkTextActor | None = None
vtk_renderer = None
text_actor = None
frames: np.ndarray = None


class vtkMatrix4x4Customized(vtk.vtkMatrix4x4):  # noqa: N801
    def __init__(self, mat: np.ndarray | None = None) -> None:
        super().__init__()
        if mat is not None and isinstance(mat, np.ndarray):
            self.SetElements(mat)

    def SetElements(self, mat: np.ndarray) -> None:
        if mat.shape != (4, 4):
            msg = "Matrix must be 4x4"
            raise ValueError(msg)
        for i in range(4):
            for j in range(4):
                self.SetElement(i, j, mat[i, j])


def _update_frame(idx: int) -> None:
    if idx < 0 or idx >= len(frames):
        return
    rotate_data = frames[idx]
    rot_mat = R.from_quat(rotate_data, scalar_first=True).as_matrix()
    rot_mat4x4 = np.pad(rot_mat, ((0, 1), (0, 1)), "constant")
    rot_mat4x4[3, 3] = 1
    temp_matrix = vtkMatrix4x4Customized(rot_mat4x4)

    props = vtk_renderer.GetViewProps()
    for i in range(props.GetNumberOfItems()):
        prop = props.GetItemAsObject(i)
        if hasattr(prop, "SetUserMatrix"):
            prop.SetUserMatrix(temp_matrix)
    text_actor.SetInput(f"Frame: {idx}")
    vtk_render_window.Render()


def _slider_callback(
    obj: vtk.vtkSliderWidget, event: vtk.vtkObject
) -> None:  # noqa: ARG001
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
    # Double-toggle sometimes helps on Windows
    slider_widget.EnabledOff()
    slider_widget.EnabledOn()
    return slider_widget


def _init_3D_scene(board_file_name: Path, nframes: int) -> None:  # noqa: N802
    global text_actor, vtk_render_window, vtk_renderer
    data_root = Path(__file__).parent
    importer = vtk.vtkGLTFImporter()
    importer.SetFileName(Path(data_root) / board_file_name)
    importer.Update()

    # Main renderer
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
    default_camera.SetPosition(0.5, 0.5, 0.5)
    default_camera.SetFocalPoint(0, 0, 0)
    default_camera.SetViewUp(0, 0, 1)
    default_camera.SetClippingRange(0.1, 2000.0)
    vtk_renderer.SetBackground(0.2, 0.2, 0.2)
    vtk_renderer.SetBackground2(0.3, 0.3, 0.3)
    vtk_renderer.SetAmbient(1, 1, 1)
    vtk_render_window.SetSize(800, 800)

    vtk_render_window.SetWindowName("SageMotion CSV Playback Demo")

    # Add text actor to overlay renderer
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("Frame: 0")
    text_actor.GetTextProperty().SetFontSize(36)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.SetDisplayPosition(40, 700)
    overlay_renderer.AddViewProp(text_actor)

    vtk_render_window_interactor.Initialize()
    vtk_render_window.Render()

    # --- SLIDER: Must be created AFTER window/interactor/overlay fully initialized! ---
    slider_widget = _add_slider_widget(vtk_render_window_interactor, nframes)
    # Trick: Sometimes toggling helps
    slider_widget.EnabledOff()
    slider_widget.EnabledOn()
    vtk_render_window.Render()

    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(200, 200, 200)  # Adjust size as needed

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(vtk_render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.25, 0.25)  # (xmin, ymin, xmax, ymax) for lower-left
    widget.SetEnabled(1)
    widget.InteractiveOff()

    _update_frame(0)
    vtk_render_window_interactor.Start()


def main():
    global frames
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize CSV quaternions with VTK and a 3D board model."
    )
    parser.add_argument("csv", help="CSV file with quaternion columns")
    parser.add_argument("--sensor", help="Sensor name (suffix in Quat1_...)")
    args = parser.parse_args()

    sensor_data = pd.read_csv(args.csv)

    if args.sensor is not None:
        sensor_name = args.sensor
    else:
        sensor_name = [  # noqa: RUF015
            c.split("Quat1_")[1] for c in sensor_data.columns if c.startswith("Quat1_")
        ][0]
    columns = [f"Quat{i}_{sensor_name}" for i in range(1, 5)]
    frames = sensor_data[columns].to_numpy()
    _init_3D_scene("Node.glb", len(frames))


if __name__ == "__main__":
    main()
