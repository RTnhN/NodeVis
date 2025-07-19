import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import vtk
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D, vtkSliderWidget

vtk_render_window = None
vtk_renderer = None
text_actor = None
frames = []
sensor_name = None


class vtkMatrix4x4Customized(vtk.vtkMatrix4x4):
    def __init__(self, mat=None) -> None:
        super().__init__()
        if mat is not None and isinstance(mat, np.ndarray):
            self.SetElements(mat)

    def SetElements(self, mat):
        for i in range(4):
            for j in range(4):
                self.SetElement(i, j, mat[i, j])


def update_frame(idx):
    global frames, vtk_render_window, text_actor, vtk_renderer
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


def slider_callback(obj, event):
    idx = int(obj.GetRepresentation().GetValue())
    update_frame(idx)


def add_slider_widget(iren, nframes):
    sliderRep = vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(0)
    sliderRep.SetMaximumValue(nframes - 1)
    sliderRep.SetValue(0)
    sliderRep.SetTitleText("Frame")
    sliderRep.SetLabelFormat("%.0f")
    sliderRep.GetSliderProperty().SetColor(1, 0, 0)
    sliderRep.GetTitleProperty().SetColor(1, 1, 1)
    sliderRep.GetLabelProperty().SetColor(1, 1, 1)
    sliderRep.GetSelectedProperty().SetColor(0, 1, 0)
    sliderRep.GetTubeProperty().SetColor(1, 1, 1)
    sliderRep.GetCapProperty().SetColor(1, 1, 1)
    sliderRep.SetSliderLength(0.02)
    sliderRep.SetSliderWidth(0.04)
    sliderRep.SetEndCapLength(0.01)
    sliderRep.SetEndCapWidth(0.03)
    sliderRep.SetTubeWidth(0.008)
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(0.2, 0.08)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(0.8, 0.08)

    sliderWidget = vtkSliderWidget()
    sliderWidget.SetInteractor(iren)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToJump()
    sliderWidget.EnabledOn()
    sliderWidget.AddObserver("InteractionEvent", slider_callback)
    # Double-toggle sometimes helps on Windows
    sliderWidget.EnabledOff()
    sliderWidget.EnabledOn()
    return sliderWidget


def init_3D_scene(board_file_name, nframes):
    global vtk_render_window, text_actor, vtk_renderer
    data_root = os.path.dirname(__file__)
    importer = vtk.vtkGLTFImporter()
    importer.SetFileName(os.path.join(data_root, board_file_name))
    importer.Update()

    # Main renderer
    vtk_renderer = importer.GetRenderer()
    vtk_render_window = importer.GetRenderWindow()
    vtk_render_window.SetNumberOfLayers(
        2
    )  # <-- This must be set before adding overlay renderer!

    # Overlay renderer for slider/text
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
    default_camera.SetPosition(0.5, 0, 0)
    default_camera.SetFocalPoint(0, 0, 0)
    default_camera.SetClippingRange(0.1, 2000.0)
    vtk_renderer.SetBackground(0.2, 0.2, 0.2)
    vtk_renderer.SetBackground2(0.3, 0.3, 0.3)
    vtk_renderer.SetAmbient(1, 1, 1)
    vtk_render_window.SetSize(800, 800)

    vtk_render_window.SetWindowName("SageMotion CSV Playback Demo")

    # Add text actor to overlay renderer
    global text_actor
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("Frame: 0")
    text_actor.GetTextProperty().SetFontSize(36)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.SetDisplayPosition(40, 700)
    overlay_renderer.AddViewProp(text_actor)

    vtk_render_window_interactor.Initialize()
    vtk_render_window.Render()

    # --- SLIDER: Must be created AFTER window/interactor/overlay fully initialized! ---
    sliderWidget = add_slider_widget(vtk_render_window_interactor, nframes)
    # Trick: Sometimes toggling helps
    sliderWidget.EnabledOff()
    sliderWidget.EnabledOn()
    vtk_render_window.Render()

    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(200, 200, 200)  # Adjust size as needed

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(vtk_render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.25, 0.25)  # (xmin, ymin, xmax, ymax) for lower-left
    widget.SetEnabled(1)
    widget.InteractiveOff()

    update_frame(0)
    vtk_render_window_interactor.Start()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize CSV quaternions with VTK and a 3D board model."
    )
    parser.add_argument("csv", help="CSV file with quaternion columns")
    parser.add_argument("--sensor", help="Sensor name (suffix in Quat1_...)")
    args = parser.parse_args()

    global frames, sensor_name
    df = pd.read_csv(args.csv)

    if args.sensor is not None:
        sensor_name = args.sensor
    else:
        sensor_name = [
            c.split("Quat1_")[1] for c in df.columns if c.startswith("Quat1_")
        ][0]

    frames = df[
        [
            f"Quat1_{sensor_name}",
            f"Quat2_{sensor_name}",
            f"Quat3_{sensor_name}",
            f"Quat4_{sensor_name}",
        ]
    ].to_numpy()
    print(len(frames))
    init_3D_scene("Node.glb", len(frames))


if __name__ == "__main__":
    main()
