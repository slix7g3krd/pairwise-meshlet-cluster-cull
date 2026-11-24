import pytest
import graphics.meshlet_cluster_cull as mod

CASES = [
    ("pyramid_fence", (1,)),
    ("meshlet_bounds", (1,)),
    ("batch_order", (1,)),
    ("indirect_buffer", (1,)),
    ("lod_select", (1,)),
    ("legacy_fallback", (1,)),
    ("cull_test", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("occlusion", (1,)),
    ("metrics", (1,)),
    ("cluster_size", (1,)),
    ("depth_hiz", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
