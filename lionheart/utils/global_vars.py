# NOTE: Remember to set the newest model first in the list!
INCLUDED_MODELS = [
    "detect_cancer__002__24_06_25",
]

# Validate model names
assert all([(m.split("_"[0] in ["detect", "subtype"]) for m in INCLUDED_MODELS)])

# We have the functionality for subtyping (multiclass classification)
# but there's currently too little training data for the smaller
# cancer types for it to work well across datasets (it seems)
# so it's disabled for now
ENABLE_SUBTYPING = False

# Check before dump or load
JOBLIB_VERSION = "1.4.2"

REPO_URL = "https://github.com/besenbacherlab/lionheart"

PCA_TARGET_VARIANCE_OPTIONS = [0.993, 0.994, 0.995, 0.996, 0.997]
PCA_TARGET_VARIANCE_FM_OPTIONS = [0.987, 0.988, 0.989, 0.990, 0.991, 0.992, 0.993, 0.994] # for full model with +1 dataset
LASSO_C_OPTIONS = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2]
PCA_TARGET_VARIANCE_OPTIONS_STRING = (
    "`" + " ".join([str(x) for x in PCA_TARGET_VARIANCE_OPTIONS]) + "`"
)
PCA_TARGET_VARIANCE_FM_OPTIONS_STRING = (
    "`" + " ".join([str(x) for x in PCA_TARGET_VARIANCE_FM_OPTIONS]) + "`"
)
LASSO_C_OPTIONS_STRING = "`" + " ".join([str(x) for x in LASSO_C_OPTIONS]) + "`"

LABELS_TO_USE = ["0_Control(control)", "1_Cancer(cancer)"]
