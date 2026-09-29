"""Object detection model implementations."""

from .faster_rcnn import DEFAULT_FASTER_RCNN_WEIGHTS, build_faster_rcnn

__all__ = ["DEFAULT_FASTER_RCNN_WEIGHTS", "build_faster_rcnn"]
