"""FocusEdit: spatially modulated guidance for flow-based image editing."""
__version__ = "0.1.0"

def __getattr__(name):
    if name in {"FocusEditPipeline", "FocusEditOutput"}:
        from .pipeline import FocusEditPipeline, FocusEditOutput
        return {"FocusEditPipeline": FocusEditPipeline, "FocusEditOutput": FocusEditOutput}[name]
    raise AttributeError(name)
