from utils.protocol_loader import load_all_protocols, load_protocol


def test_load_protocol_resolves_vp_and_fp_specs_by_canonical_id():
    vp_spec = load_protocol("VP-01")
    fp_spec = load_protocol("FP-01")

    assert vp_spec is not None
    assert vp_spec.protocol_id == "VP-01"
    assert fp_spec is not None
    assert fp_spec.protocol_id == "FP-01"


def test_load_protocol_resolves_vp_and_fp_specs_by_runtime_alias():
    vp_spec = load_protocol("VP_07_TMSCausalInterventions")
    fp_spec = load_protocol("FP_01_ActiveInference")

    assert vp_spec is not None
    assert vp_spec.protocol_id == "VP-07"
    assert fp_spec is not None
    assert fp_spec.protocol_id == "FP-01"


def test_load_all_protocols_includes_generated_vp_and_fp_specs():
    specs = load_all_protocols()

    assert "VP-22" in specs
    assert "FP-12" in specs
