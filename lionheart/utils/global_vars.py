# NOTE: Remember to set the newest model first in the list!
INCLUDED_MODELS = [
    "detect_cancer__001__25_11_24",
]

# Validate model names
assert all([(m.split("_"[0] in ["detect", "subtype"]) for m in INCLUDED_MODELS)])

# We have the functionality for subtyping (multiclass classification)
# but there's currently too little training data for the smaller
# cancer types for it to work well across datasets (it seems)
# so it's disabled for now
ENABLE_SUBTYPING = False

# Check before dump or load
JOBLIB_VERSION = "1.2.0"

REPO_URL = "https://github.com/besenbacherlab/lionheart"

PCA_TARGET_VARIANCE_OPTIONS = [0.994, 0.995, 0.996, 0.997, 0.998]
LASSO_C_OPTIONS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
