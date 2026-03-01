"""Neural Prediction Market -- model zoo."""

from .pretrained_npm import (                      # noqa: F401
    PretrainedNPM,
    PretrainedBackbone,
    PretrainedEnsemble,
    PretrainedMCDropout,
    PretrainedMoE,
    BACKBONE_REGISTRY,
    solve_ensemble_hidden_dim,
    solve_mc_hidden_dim,
    solve_moe_hidden_dim,
)

from .uq_heads import (                            # noqa: F401
    PretrainedSNGP,
    PretrainedDUE,
    PretrainedDUQ,
    PretrainedEvidential,
    SNGPHead,
    DUEHead,
    DUQHead,
    EvidentialHead,
    edl_mse_loss,
    edl_digamma_loss,
)

from .posthoc_ood import (                         # noqa: F401
    energy_score,
    odin_score,
    MahalanobisDetector,
    ViMDetector,
    ReActDetector,
    KNNDetector,
    run_all_posthoc_ood,
)
