"""Backward-compatible import path: ``from pipeline.cvpr23_cnn_lstm import CNNLSTM``.

Implementation is in :mod:`pipeline.cnn_lstm_cvpr23`. If Colab still raises
``ImportError: cannot import name 'CNNLSTM'``, the Drive copy of *this* file may
be empty (placeholder); use ``from pipeline.cnn_lstm_cvpr23 import CNNLSTM``
and ensure ``pipeline/cnn_lstm_cvpr23.py`` is synced to Drive.
"""

from pipeline.cnn_lstm_cvpr23 import CNNLSTM

__all__ = ["CNNLSTM"]
