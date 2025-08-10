
# Set active dataset
CURRENT_DATASET = 'IMSLICE'
if CURRENT_DATASET == 'MISD':
    from .config import MISD_CONFIG as DATASET_CONFIG
elif CURRENT_DATASET == '4CAM':
    from .config import CAM4_CONFIG as DATASET_CONFIG
elif CURRENT_DATASET == 'IMSLICE':
    from .config import IMSLICE_CONFIG as DATASET_CONFIG
else:
    raise ValueError(f"Unknown dataset: {CURRENT_DATASET}")
