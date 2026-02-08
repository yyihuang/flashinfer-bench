"""
Test GDN decode k-last reference implementation against FlashInfer kernel.

Run with:
    pytest test_gdn_decode_qk4_v8_d128_k_last.py -v
    python test_gdn_decode_qk4_v8_d128_k_last.py
"""

import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"
WORKLOAD_INPUTS = {
    "q": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "q",
    },
    "k": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "k",
    },
    "v": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "v",
    },
    "state": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "state",
    },
    "A_log": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "A_log",
    },
    "a": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "a",
    },
    "dt_bias": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "dt_bias",
    },
    "b": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "b",
    },
    "scale": {
        "type": "scalar",
        "value": 0.08838834764831843,
    },
    "cu_seqlens": {
        "type": "safetensors",
        "path": "./blob/workloads/gdn/gdn_decode_qk4_v8_d128_k_last_8add0e58-121b-4d25-b450-cb03e665511c.safetensors",
        "tensor_key": "cu_seqlens",
    },
}


def _resolve_real_safetensors_path(workload_path: str) -> Path:
    """Resolve the real safetensors path from the workload path UUID."""
    match = re.search(
        r"gdn_decode_qk4_v8_d128_k_last_([0-9a-f-]+)\.safetensors", workload_path
    )
    if not match:
        raise ValueError(f"Unable to extract UUID from workload path: {workload_path}")
    uuid = match.group(1)
    return Path("/home/averyh/sglang/tmp") / f"gdn_decode_qk4_v8_d128_k_last_{uuid}.safetensors"


def load_workload_inputs(device: str = "cuda"):
    """Load workload tensors from the real safetensors file."""
    any_path = next(iter(WORKLOAD_INPUTS.values()))["path"]
    real_path = _resolve_real_safetensors_path(any_path)
    # Load on CPU first, then move to target device to avoid safetensors device errors.
    tensors = load_file(str(real_path), device="cpu")

    inputs = {}
    for key, spec in WORKLOAD_INPUTS.items():
        if key == "cu_seqlens":
            continue
        if spec["type"] == "scalar":
            inputs[key] = float(spec["value"])
            continue
        tensor = tensors[spec["tensor_key"]]
        inputs[key] = tensor.to(device)
    return inputs


def load_definition(name: str) -> Definition:
    """Load a definition by name from definitions directory."""
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    """Compile reference implementation to callable function."""
    namespace = {"torch": torch, "math": math, "F": F}
    exec(reference_code, namespace)
    return namespace["run"]


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN decode requires SM90+ or SM100+, but got SM{cc[0]}{cc[1]}")


def run_kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Run FlashInfer kernel (pretranspose version uses k-last layout)."""
    B, T, num_q_heads, K = q.shape
    num_v_heads = v.shape[2]

    # Pre-allocate output
    output = torch.empty(B, T, num_v_heads, K, dtype=q.dtype, device=q.device)

    # Call kernel
    out, new_state = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state.clone(),
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        scale=scale,
        output=output,
        use_qk_l2norm=False,
    )

    return out, new_state


def test_correctness(atol=5e-3, rtol=5e-3):
    """Test correctness of reference implementation against FlashInfer."""
    _skip_if_not_sm90_or_later()

    inputs = load_workload_inputs(device="cuda")
    batch_size = inputs["q"].shape[0]

    print(f"\n{'='*60}")
    print(f"Testing GDN decode k-last, batch_size={batch_size}")
    print(f"{'='*60}")

    # Load definition and compile reference
    definition = load_definition("gdn_decode_qk4_v8_d128_k_last")
    run = compile_reference(definition.reference)

    # Run reference from definition
    print("Running reference implementation from definition...")
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )
    ref_output, ref_new_state = ref_result

    # Run kernel
    print("Running FlashInfer kernel...")
    kernel_output, kernel_new_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )

    # Compare outputs
    print("\nComparing outputs...")

    ref_o_f32 = ref_output.float()
    kernel_o_f32 = kernel_output.float()

    # Absolute difference metrics
    abs_diff_o = torch.abs(ref_o_f32 - kernel_o_f32)
    max_abs_diff_o = abs_diff_o.max().item()
    mean_abs_diff_o = abs_diff_o.mean().item()

    # Relative difference metrics (avoid division by zero)
    rel_diff_o = abs_diff_o / (torch.abs(ref_o_f32) + 1e-10)
    max_rel_diff_o = rel_diff_o.max().item()
    mean_rel_diff_o = rel_diff_o.mean().item()

    # Cosine similarity
    ref_flat = ref_o_f32.reshape(-1)
    kernel_flat = kernel_o_f32.reshape(-1)
    cosine_sim_o = F.cosine_similarity(ref_flat.unsqueeze(0), kernel_flat.unsqueeze(0)).item()

    # Mean Squared Error
    mse_o = ((ref_o_f32 - kernel_o_f32) ** 2).mean().item()

    print("\nOutput tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_o:.6e}")
    print(f"  Max relative difference: {max_rel_diff_o:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_o:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_o:.6e}")
    print(f"  Cosine similarity: {cosine_sim_o:.6f}")
    print(f"  MSE: {mse_o:.6e}")

    # State comparison
    abs_diff_s = torch.abs(ref_new_state - kernel_new_state)
    max_abs_diff_s = abs_diff_s.max().item()
    mean_abs_diff_s = abs_diff_s.mean().item()

    # State relative difference
    rel_diff_s = abs_diff_s / (torch.abs(ref_new_state) + 1e-10)
    max_rel_diff_s = rel_diff_s.max().item()
    mean_rel_diff_s = rel_diff_s.mean().item()

    # State cosine similarity
    ref_state_flat = ref_new_state.reshape(-1)
    kernel_state_flat = kernel_new_state.reshape(-1)
    cosine_sim_s = F.cosine_similarity(
        ref_state_flat.unsqueeze(0), kernel_state_flat.unsqueeze(0)
    ).item()

    # State MSE
    mse_s = ((ref_new_state - kernel_new_state) ** 2).mean().item()

    print("\nState tensor comparison:")
    print(f"  Max absolute difference: {max_abs_diff_s:.6e}")
    print(f"  Max relative difference: {max_rel_diff_s:.6e}")
    print(f"  Mean absolute difference: {mean_abs_diff_s:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff_s:.6e}")
    print(f"  Cosine similarity: {cosine_sim_s:.6f}")
    print(f"  MSE: {mse_s:.6e}")

    output_close = torch.allclose(ref_o_f32, kernel_o_f32, atol=atol, rtol=rtol)
    state_close = torch.allclose(ref_new_state, kernel_new_state, atol=atol, rtol=rtol)

    if output_close and state_close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
        return True
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
        return False


@pytest.mark.parametrize("batch_size", [1])
def test_gdn_decode_k_last(batch_size: int):
    """Pytest test using captured workload inputs."""
    _skip_if_not_sm90_or_later()

    # Load definition and compile reference
    definition = load_definition("gdn_decode_qk4_v8_d128_k_last")
    run = compile_reference(definition.reference)

    inputs = load_workload_inputs(device="cuda")
    if inputs["q"].shape[0] != batch_size:
        pytest.skip(f"Workload batch_size={inputs['q'].shape[0]} does not match {batch_size}")

    # Run reference from definition
    ref_result = run(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )
    ref_output, ref_new_state = ref_result

    # Run kernel
    kernel_output, kernel_new_state = run_kernel(
        inputs["q"].clone(),
        inputs["k"].clone(),
        inputs["v"].clone(),
        inputs["state"].clone(),
        inputs["A_log"].clone(),
        inputs["a"].clone(),
        inputs["dt_bias"].clone(),
        inputs["b"].clone(),
        inputs["scale"],
    )

    atol, rtol = 1e-2, 1e-2

    torch.testing.assert_close(
        kernel_output,
        ref_output,
        atol=atol,
        rtol=rtol,
        msg=f"Output mismatch for batch_size={batch_size}",
    )
    torch.testing.assert_close(
        kernel_new_state,
        ref_new_state,
        atol=atol,
        rtol=rtol,
        msg=f"State mismatch for batch_size={batch_size}",
    )

    print(f"✓ GDN decode k-last test passed (batch_size={batch_size})")


def main():
    """Run tests."""
    print("Testing GDN Decode K-Last Reference Implementation")
    print(
        "Loading definition from: flashinfer_trace/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json"
    )

    test_configs = [1]

    passed = 0
    total = len(test_configs)

    for batch_size in test_configs:
        try:
            if test_correctness():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
