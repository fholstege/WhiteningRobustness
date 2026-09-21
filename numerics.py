"""Central numerical-dtype policy for the evaluation pipeline.

Stored real-data representations and their preprocessing intentionally remain
float32. Logistic optimization uses an exact float64 view of those float32
values so tighter solver tolerances do not run into float32 line-search
precision. Keeping these stages separate changes optimization arithmetic,
not the fitted representation transform or its covariance estimate. Synthetic
experiments instead use float64 from generation through evaluation.
"""

from __future__ import annotations

import numpy as np


# Artifact and representation-transform contract.
STORED_EMBEDDING_DTYPE = np.float32
PREPROCESSING_DTYPE = np.float32
SPECTRAL_ESTIMATION_DTYPE = np.float64


# Last-layer estimation and evaluation. These remain separate settings so the
# policy is explicit if one stage is changed independently in the future.
OPTIMIZATION_DTYPE = np.float64
MODEL_PARAMETER_DTYPE = np.float64
METRIC_DTYPE = np.float32

# Synthetic experiments retain precision through generation and preprocessing.
SYNTHETIC_DTYPE = np.float64
