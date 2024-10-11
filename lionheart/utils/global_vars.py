# NOTE: Remember to set the newest model first in the list!
INCLUDED_MODELS = [
    "detect_cancer__007__20_09_24",
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
