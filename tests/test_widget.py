from napari_4dflow._widget import UnwrapWidget


def test_widget_creation(make_napari_viewer):
    widget = UnwrapWidget(make_napari_viewer())
    assert widget is not None
