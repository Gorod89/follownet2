"""Leaderboard of the benchmark (milestone 5, docs/leaderboard.md).

* :mod:`~follownet2.leaderboard.checklist`: the checklist gate (:func:`validate_entry`);
* :mod:`~follownet2.leaderboard.tables`: main, driver-leak, transfer and refused tables, provenance;
* :mod:`~follownet2.leaderboard.render`: Markdown and the self-contained HTML page;
* :mod:`~follownet2.leaderboard.stability`: the rank-stability plot.
"""

from follownet2.leaderboard.checklist import DRIVER_LEAK, entry_label, validate_dataset, validate_entry
from follownet2.leaderboard.tables import Board, build_board, build_leaderboard

__all__ = ["DRIVER_LEAK", "Board", "build_board", "build_leaderboard", "entry_label", "validate_dataset", "validate_entry"]
