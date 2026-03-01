from __future__ import annotations

import torch


def causal_iir_filter_parallel(
    charges: torch.Tensor,
    lambdas: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    time_dim: int = 1,
) -> torch.Tensor:
    """
    Fast causal IIR filter using a parallel prefix scan formulation.

    Recurrence:
        Phi_t = lambda * Phi_{t-1} + C_t

    Args:
        charges: Tensor with shape [..., seq_len, field_dim] after moving `time_dim`.
        lambdas: [field_dim] decay values in (0, 1), one per field dimension.
        initial_state: Optional tensor [..., field_dim] representing Phi_{-1}.
        time_dim: Dimension index of sequence length in `charges`.

    Returns:
        Tensor with same shape as `charges` containing Phi for each timestep.
    """
    if charges.ndim < 2:
        raise ValueError(f"charges must be at least 2D, got shape {tuple(charges.shape)}.")

    normalized_time_dim = time_dim % charges.ndim
    if normalized_time_dim == charges.ndim - 1:
        raise ValueError("time_dim cannot be the last dimension; last dim is field_dim.")

    x = charges.movedim(normalized_time_dim, -2)  # x:[..., seq_len, field_dim]
    seq_len = x.shape[-2]
    field_dim = x.shape[-1]

    if lambdas.ndim != 1 or lambdas.shape[0] != field_dim:
        raise ValueError(
            f"lambdas must have shape [{field_dim}], got {tuple(lambdas.shape)}."
        )
    if torch.any((lambdas <= 0.0) | (lambdas >= 1.0)):
        raise ValueError("All lambda values must be in the open interval (0, 1).")

    if initial_state is not None:
        expected_state_shape = x.shape[:-2] + (field_dim,)
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}."
            )
        init = initial_state.to(device=x.device, dtype=x.dtype)
    else:
        init = None

    # Each timestep is an affine map y -> a*y + b with:
    # a=lambda, b=C_t.
    # We run a Hillis-Steele parallel prefix scan on (a, b) using:
    # (a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1).
    lambda_values = lambdas.to(device=x.device, dtype=x.dtype).view(
        (1,) * (x.ndim - 2) + (1, field_dim)
    )
    a_scan = lambda_values.expand(x.shape).clone()  # a_scan:[..., seq_len, field_dim]
    b_scan = x.clone()  # b_scan:[..., seq_len, field_dim]

    offset = 1
    while offset < seq_len:
        a_prev = a_scan.clone()
        b_prev = b_scan.clone()
        a_scan[..., offset:, :] = a_prev[..., offset:, :] * a_prev[..., :-offset, :]
        b_scan[..., offset:, :] = b_prev[..., offset:, :] + a_prev[..., offset:, :] * b_prev[..., :-offset, :]
        offset <<= 1

    filtered = b_scan
    if init is not None:
        filtered = filtered + a_scan * init.unsqueeze(-2)

    return filtered.movedim(-2, normalized_time_dim)


def causal_iir_filter_sequential_reference(
    charges: torch.Tensor,
    lambdas: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    time_dim: int = 1,
) -> torch.Tensor:
    """
    Slow reference implementation with a Python for-loop over timesteps.
    Intended only for verification tests.
    """
    if charges.ndim < 2:
        raise ValueError(f"charges must be at least 2D, got shape {tuple(charges.shape)}.")

    normalized_time_dim = time_dim % charges.ndim
    if normalized_time_dim == charges.ndim - 1:
        raise ValueError("time_dim cannot be the last dimension; last dim is field_dim.")

    x = charges.movedim(normalized_time_dim, -2)  # x:[..., seq_len, field_dim]
    seq_len = x.shape[-2]
    field_dim = x.shape[-1]

    if lambdas.ndim != 1 or lambdas.shape[0] != field_dim:
        raise ValueError(
            f"lambdas must have shape [{field_dim}], got {tuple(lambdas.shape)}."
        )
    if torch.any((lambdas <= 0.0) | (lambdas >= 1.0)):
        raise ValueError("All lambda values must be in the open interval (0, 1).")

    lambda_values = lambdas.to(device=x.device, dtype=x.dtype).view(
        (1,) * (x.ndim - 2) + (field_dim,)
    )
    prev = (
        torch.zeros(x.shape[:-2] + (field_dim,), device=x.device, dtype=x.dtype)
        if initial_state is None
        else initial_state.to(device=x.device, dtype=x.dtype)
    )

    out = torch.empty_like(x)
    for t in range(seq_len):
        prev = lambda_values * prev + x[..., t, :]
        out[..., t, :] = prev
    return out.movedim(-2, normalized_time_dim)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = _pick_device()
    torch.manual_seed(7)

    # Test 1: Standard [batch, seq_len, D] layout with initial state.
    batch_size, seq_len, field_dim = 3, 257, 8
    charges = torch.randn(batch_size, seq_len, field_dim, device=device)
    lambdas = torch.rand(field_dim, device=device) * 0.93 + 0.03  # strictly in (0, 1)
    initial_state = torch.randn(batch_size, field_dim, device=device)

    fast = causal_iir_filter_parallel(
        charges,
        lambdas,
        initial_state=initial_state,
        time_dim=1,
    )
    slow = causal_iir_filter_sequential_reference(
        charges,
        lambdas,
        initial_state=initial_state,
        time_dim=1,
    )

    # Test 2: Extra leading dimensions to confirm broadcasting semantics.
    charges_nd = torch.randn(2, 4, seq_len, field_dim, device=device)
    initial_state_nd = torch.randn(2, 4, field_dim, device=device)
    fast_nd = causal_iir_filter_parallel(
        charges_nd,
        lambdas,
        initial_state=initial_state_nd,
        time_dim=2,
    )
    slow_nd = causal_iir_filter_sequential_reference(
        charges_nd,
        lambdas,
        initial_state=initial_state_nd,
        time_dim=2,
    )

    atol = 1e-5
    rtol = 1e-4
    max_abs_err = max(
        (fast - slow).abs().max().item(),
        (fast_nd - slow_nd).abs().max().item(),
    )

    if not (torch.allclose(fast, slow, atol=atol, rtol=rtol) and torch.allclose(fast_nd, slow_nd, atol=atol, rtol=rtol)):
        raise AssertionError(
            "Parallel and sequential IIR filters differ: "
            f"max_abs_err={max_abs_err:.6e}, atol={atol}, rtol={rtol}."
        )

    print(
        "PASS: parallel causal IIR filter matches sequential reference "
        f"on device={device} (max_abs_err={max_abs_err:.6e})."
    )
