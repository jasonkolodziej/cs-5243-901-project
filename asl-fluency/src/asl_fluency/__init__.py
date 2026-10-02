"""ASL Fingerspelling Fluency Scoring.

Companion code for the research proposal:
"Automated Fluency and Speed Scoring for ASL Fingerspelling".

Pipeline stages:
    data         -> ChicagoFSWild(+) metadata + human fluency ratings
    landmarks    -> per-frame MediaPipe hand-landmark extraction
    segmentation -> letter-boundary estimation within a clip
    features     -> kinematic fluency features (rate, smoothness, pauses)
    models       -> naive baseline + learned fluency scorers
    evaluation   -> correlation / MAE / ablations / inter-rater agreement
"""

__version__ = "0.1.0"
