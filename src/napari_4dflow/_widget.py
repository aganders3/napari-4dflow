from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from .unwrap import unwrap

if TYPE_CHECKING:
    import napari


@magicgui
def unwrap_real_imaginary_two_layers(
    real: "napari.types.ImageData",
    imaginary: "napari.types.ImageData",
) -> "napari.types.ImageData":
    phase = np.angle(real + 1j * imaginary)
    return phase + 2 * np.pi * unwrap(phase)


@magicgui
def unwrap_real_imaginary_one_layer(
    data: "napari.types.ImageData",
    real_imag_axis: int,
) -> "napari.types.ImageData":
    assert real_imag_axis in range(data.ndim), "invalid axis"
    real = data.take([0], real_imag_axis)
    imaginary = data.take([1], real_imag_axis)
    phase = np.angle(real + 1j * imaginary)
    return phase + 2 * np.pi * unwrap(phase)


@magicgui
def unwrap_phase_one_layer(
    data: "napari.types.ImageData",
    mag_phase_axis: int,
) -> "napari.types.ImageData":
    assert mag_phase_axis in range(data.ndim), "invalid axis"
    phase = data.take([-1], mag_phase_axis)
    return phase + 2 * np.pi * unwrap(phase)


@magicgui
def unwrap_phase_only_layer(
    phase: "napari.types.ImageData",
) -> "napari.types.ImageData":
    return phase + 2 * np.pi * unwrap(phase)


_SUPPORTED_DATA_TYPES = {
    "real/imaginary (two layers)": unwrap_real_imaginary_two_layers,
    "real/imaginary (one layer)": unwrap_real_imaginary_one_layer,
    "magnitude/phase (one layer)": unwrap_phase_one_layer,
    "phase only (one layer)": unwrap_phase_only_layer,
}


class UnwrapWidget(QWidget):
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._selection_widget = widgets.ComboBox(
            choices=list(_SUPPORTED_DATA_TYPES.keys()),
            value=list(_SUPPORTED_DATA_TYPES.keys())[0],
            label="Select data type",
        )
        self._function_widgets = QStackedWidget(self)
        for widget in _SUPPORTED_DATA_TYPES.values():
            self._function_widgets.addWidget(widget.native)

        self._layout = QVBoxLayout()
        self._layout.addWidget(self._selection_widget.native)
        self._layout.addWidget(self._function_widgets)
        self.setLayout(self._layout)

        self._selection_widget.changed.connect(self._on_selection_changed)
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

    def _on_selection_changed(self, value=None):
        new_widget = _SUPPORTED_DATA_TYPES[value]
        self._function_widgets.setCurrentWidget(new_widget.native)
        self.update()

    def _on_layers_changed(self, event):
        for widget in _SUPPORTED_DATA_TYPES.values():
            widget.reset_choices()


class AngiogramWidget(QWidget):
    # TODO: select mag and velocity layers -> create angiogram layer
    pass


class FlowVectorsWidget(QWidget):
    # TODO: select flow directions + mask -> create vectors layer
    # TODO: add mask methods
    pass
