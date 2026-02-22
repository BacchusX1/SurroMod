"""
Data Splitter
=============
Decomposes multi-dimensional data into separate streams along one or more axes.

Supported split modes (matching the frontend ``SplitMode`` type):

+----------------+----------------------------+-----------------------------------+
| ``split_mode`` | Input shape assumed        | Output per port                   |
+================+============================+===================================+
| ``channel``    | ``(N, C, H, W)``           | C  × ``(N, H, W)``               |
+----------------+----------------------------+-----------------------------------+
| ``channel_x``  | ``(N, C, H, W)``           | C·W × ``(N, H)``  (channel × x)  |
+----------------+----------------------------+-----------------------------------+
| ``channel_y``  | ``(N, C, H, W)``           | C·H × ``(N, W)``  (channel × y)  |
+----------------+----------------------------+-----------------------------------+
| ``x``          | ``(N, C, H, W)``           | W  × ``(N, C, H)`` (x-slices)    |
+----------------+----------------------------+-----------------------------------+
| ``y``          | ``(N, C, H, W)``           | H  × ``(N, C, W)`` (y-slices)    |
+----------------+----------------------------+-----------------------------------+

The number of output ports (``n_outputs``) is set in the frontend and determines
how many ``channel-*`` source handles the node exposes.  When the actual number
of slices produced by the split mode differs from ``n_outputs``, slices are
grouped or padded to match.
"""


class DataSplitter:
    """Split multi-dimensional data along configurable axes into separate streams."""
    pass
