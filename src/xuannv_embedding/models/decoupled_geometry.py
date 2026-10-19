"""Keep coarse semantic gradients out of optional local detail paths."""

from xuannv_embedding.models.time_modulated import TimeModulatedFusion


class PublicSemanticGeometryFusion(TimeModulatedFusion):
    def __init__(self, **kwargs):
        super().__init__(semantic_public_only=True, **kwargs)
        self.collect_local_geometry = True
