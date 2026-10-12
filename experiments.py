"""Named experiment configurations (overrides of train.DEFAULTS) used in the paper.

The names are those of the released checkpoint folders (checkpoints/<experiment>/<pair>_s<seed>.pt).
Suffix _traj = trajectory only; _mapf = with SingularTrajectory's scene-map anchor adjustment
(mirror augmentation kept; mirrored anchors are mirrored back before the map lookup).

    experiment                         paper
    full_traj_lam03 / full_mapf_lam03  ST-CRF (trajectory only / with scene map): Tables 3-7
    none_traj / none_mapf              w/o step-intention: Tables 4 and 7, RQ5
    fullce_traj_l03 / fullce_mapf_l03  CE (hist. + pred.): Table 4, RQ5
    obscrf_traj_l03 / obscrf_mapf_l03  CRF (hist. only): Table 4, RQ5
    crf2_traj_lam03 / crf2_mapf_l03    CRF (pred. only): Table 4, RQ5
    stoch_*                            multi-modal (S = 20, best-of-20) leave-one-out: Table 7
"""
EXPERIMENTS = {
    # ST-CRF: one CRF chain over the observed steps 2..8 and the 12 predicted steps; the marginals of the
    # predicted steps drive the intention-driven anchor
    'full_traj_lam03': dict(intent='full_crf', use_map=False, avg2=True, lam=0.3),
    'full_mapf_lam03': dict(intent='full_crf', use_map=True, avg2=True, map_flip=True, lam=0.3),
    # without step-intention supervision: mean anchor of the training futures, as SingularTrajectory
    'none_traj': dict(intent='none', use_map=False),
    'none_mapf': dict(intent='none', use_map=True, map_flip=True),
    # cross-entropy instead of the CRF loss, same labels as ST-CRF
    'fullce_traj_l03': dict(intent='full_ce', use_map=False, avg2=True, lam=0.3),
    'fullce_mapf_l03': dict(intent='full_ce', use_map=True, avg2=True, map_flip=True, lam=0.3),
    # CRF over the observed steps only, auxiliary loss with the mean anchor
    'obscrf_traj_l03': dict(intent='obs_crf', use_map=False, lam=0.3),
    'obscrf_mapf_l03': dict(intent='obs_crf', use_map=True, map_flip=True, lam=0.3),
    # CRF over the predicted steps only, driving the anchor
    'crf2_traj_lam03': dict(intent='crf', use_map=False, avg2=True, lam=0.3),
    'crf2_mapf_l03': dict(intent='crf', use_map=True, avg2=True, map_flip=True, lam=0.3),
    # multi-modal leave-one-out (S = 20 k-means prototypes, best-of-20)
    'stoch_full_traj': dict(intent='full_crf', use_map=False, avg2=True, lam=0.3, num_samples=20),
    'stoch_full_mapf': dict(intent='full_crf', use_map=True, avg2=True, map_flip=True, lam=0.3, num_samples=20),
    'stoch_none_traj': dict(intent='none', use_map=False, num_samples=20),
    'stoch_none_mapf': dict(intent='none', use_map=True, map_flip=True, num_samples=20),
}
