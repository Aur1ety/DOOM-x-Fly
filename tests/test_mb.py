"""Regression guards for the mushroom-body memory modules, on a tiny synthetic wiring cache (no data needed).

Each test pins a defect that an adversarial review found in these modules, so it cannot come back silently:
the k-winners pool, the exact one-pairing calibration, the generalisation table being read from the ONE-pairing
memory (not the saturated end of the pairing curve), nan drops serialised as null, the reciprocal T-maze index
being 0 by construction when untrained, the two-hop relay set excluding MBONs and DNs, the timed rule's closed
form, the leave-one-out decoder scoring a separable code at 1 (the old one-sample decoder was identically 0),
and the lesion battery's baseline-corrected behavioural readout.

    python -m pytest tests/test_mb.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from flybrain.eval import mb_olfactory
from flybrain.eval.mb_olfactory import MushroomBody, drop, json_safe

N_GLOM, PN_PER_GLOM, N_KC, N_DRIVABLE = 16, 3, 80, 72
MBON_TYPES = ["MBON11", "MBON11", "MBON03", "MBON03", "MBON05", "MBON10"]


@pytest.fixture(scope="module")
def wiring(tmp_path_factory):
    """A mb_build-shaped cache: 48 PNs in 16 glomeruli (enough for the modules' six-glomerulus odours and their
    generalisation pools), 80 KCs (72 with PN input, 12 of them with visual input), 6 MBONs of 4 types, PPL101
    onto MBON11 only, PAM onto MBON03/05, 5 DNs, 2 APL cells."""
    rng = np.random.default_rng(0)
    n_pn = N_GLOM * PN_PER_GLOM
    opn_glom = np.array([f"G{i // PN_PER_GLOM:02d}" for i in range(n_pn)])
    kc_type = np.array(["KCg-m"] * 40 + ["KCab-s"] * 20 + ["KCg-d"] * 20)
    W_pk = np.zeros((N_KC, n_pn))
    for k in range(N_DRIVABLE):
        W_pk[k, rng.choice(n_pn, 4, replace=False)] = rng.integers(5, 20, 4)
    vpn_type = np.array(["LC4", "LC4", "LPLC2", "LPLC2", "MeVP1", "MeVP1"])
    W_vk = np.zeros((N_KC, 6))
    for k in range(60, 72):
        W_vk[k, rng.choice(6, 2, replace=False)] = rng.integers(1, 5, 2)
    km = np.zeros((len(MBON_TYPES), N_KC)); km[:, :N_DRIVABLE] = rng.integers(1, 6, (len(MBON_TYPES), N_DRIVABLE))
    km = sparse.coo_matrix(km)
    dan_type = np.array(["PPL101", "PPL101", "PAM01", "PAM04"])
    dan_mbon = np.zeros((len(MBON_TYPES), 4)); dan_mbon[0:2, 0:2] = 10.0; dan_mbon[2:5, 2:4] = 8.0
    dn_type = np.array(["DNa02", "DNa03", "DNp01", "DNb01", "DNg01"])
    W1 = rng.integers(0, 3, (5, len(MBON_TYPES))).astype(float); W2 = rng.integers(0, 10, (5, len(MBON_TYPES))).astype(float)
    mbon_in_total = np.asarray(km.sum(1)).ravel() * 4.0            # KC input is exactly a quarter of each MBON's input
    W_pk = sparse.csr_matrix(W_pk); W_vk = sparse.csr_matrix(W_vk)
    path = tmp_path_factory.mktemp("mb") / "mb_wiring.npz"
    np.savez(path, kc_type=kc_type.astype("U16"), mbon_type=np.array(MBON_TYPES).astype("U16"), opn_glom=opn_glom.astype("U16"),
             vpn_type=vpn_type.astype("U24"), dan_type=dan_type.astype("U16"),
             pk_data=W_pk.data, pk_indices=W_pk.indices, pk_indptr=W_pk.indptr, pk_shape=np.array(W_pk.shape),
             vk_data=W_vk.data, vk_indices=W_vk.indices, vk_indptr=W_vk.indptr, vk_shape=np.array(W_vk.shape),
             km_row=km.row, km_col=km.col, km_data=km.data, km_shape=np.array(km.shape),
             dan_mbon=dan_mbon, dn_type=dn_type.astype("U16"), dn_mbon_direct=W1, dn_mbon_2hop=W2,
             kc_apl=np.full((2, N_KC), 40.0), apl_kc=np.full((N_KC, 2), 40.0),
             mbon_in_total=mbon_in_total, kc_body=np.arange(N_KC) + 1000,
             build_time=np.array("test"), build_source_sha=np.array("deadbeef"))
    return path


@pytest.fixture(scope="module")
def neurons(tmp_path_factory):
    """neurons.parquet stand-in: the MBON types with a transmitter, as build_valence reads it."""
    df = pd.DataFrame({"bodyId": [1, 2, 3, 4, 5, 6], "type": MBON_TYPES,
                       "consensus_nt": ["gaba", "gaba", "glutamate", "glutamate", "glutamate", "gaba"]})
    path = tmp_path_factory.mktemp("g") / "neurons.parquet"; df.to_parquet(path); return path


def mb_of(wiring, **kw):
    return MushroomBody(None, None, wiring=wiring, binary=True, **kw)


def test_kwta_runs_over_the_drivable_pool_only(wiring):
    mb = mb_of(wiring)
    assert mb.n_drivable_kc == N_DRIVABLE
    code = mb.kc_code(mb.odour([0, 1, 2]))
    assert int((code > 0).sum()) == max(1, round(0.05 * N_DRIVABLE))
    assert not code[N_DRIVABLE:].any()                                        # cells without PN input never fire
    vis = mb_of(wiring, modality="visual")
    assert vis.n_drivable_kc == 12                                             # the visual pool is the VPN-driven cells
    assert int((vis.kc_code(vis.odour([0, 1])) > 0).sum()) <= 1


def test_one_calibrated_pairing_hits_the_target_exactly(wiring):
    mb = mb_of(wiring); code = mb.kc_code(mb.odour([0, 1, 2]))
    mask = mb.compartment_mask("punish")
    assert mask.tolist() == [True, True, False, False, False, False]          # PPL101 lands on MBON11 only
    assert mb.compartment_mask("reward").tolist() == [False, False, True, True, True, False]
    mb.calibrate_lr(code, mask, 0.9); mb.reset()
    before = mb.mbon_response(code); mb.reinforce(code, "punish")
    assert abs(drop(before, mb.mbon_response(code), mask) - 0.9) < 1e-6
    mb.reset(); assert torch.equal(mb.W, torch.ones(mb.n_edges))             # reset restores every weight


def test_generalisation_is_read_from_the_one_pairing_memory(wiring):
    """The review found the generalisation table measured after the 8-pairing curve while labelled 'one pairing'.
    With a binary code and one calibrated pairing, drop == 0.9 * overlap exactly, and W bottoms at 0.1 not 0.1**8."""
    a = mb_olfactory.build_parser().parse_args(["--binary-code", "--wiring", str(wiring), "--odours", "4",
                                                 "--glom-per-odour", "3", "--gen-reps", "5", "--curve-max", "8"])
    res = json_safe(mb_olfactory.run(a))
    assert res["rule"]["pairings"] == 1
    assert abs(res["one_memory"]["W_stats"]["min"] - 0.1) < 1e-6
    for row in res["generalisation"].values():
        assert abs(row["drop_MBON11"] - 0.9 * row["overlap_at_MBON11"]) < 1e-3
    assert res["drop_vs_pairings_MBON11"][8]["paired_A_MBON11"] > 0.99       # the curve itself still saturates
    assert res["circuit"]["n_drivable_KC"] == N_DRIVABLE and res["circuit"]["kwta_k"] == round(0.05 * N_DRIVABLE)
    assert abs(res["circuit"]["MBON11_input_frac_from_KC"] - 0.25) < 1e-6


def test_nan_drop_is_serialised_as_null():
    z = torch.zeros(3); mask = torch.tensor([True, False, False])
    d = drop(z, z, mask); assert np.isnan(d)                                   # nan when the masked cells carry no drive
    assert json.loads(json.dumps(json_safe({"x": [d, 1.0]}), allow_nan=False)) == {"x": [None, 1.0]}


def test_wiring_provenance_is_loaded(wiring):
    mb = mb_of(wiring)
    assert mb.wiring_build == {"build_time": "test", "build_source_sha": "deadbeef"}
    assert len(mb.kc_body) == N_KC and mb.input_totals_source.startswith("full connectome")


def test_tmaze_untrained_is_zero_by_construction_and_learning_shifts_choice(wiring, neurons):
    from flybrain.eval import mb_behaviour
    a = argparse.Namespace(subgraph=None, neurons=neurons, wiring=wiring, modality="olfactory", sparsity=0.05, odours=4,
                           glom_per_odour=3, pairings=1, lr="auto", target_drop=0.9, strength=1.0, recover_rate=1.0,
                           w_max=2.0, binary_code=True, shuffle=None, punish=["PPL101"], reward=["PAM"],
                           betas=[4.0, 8.0], pi_seeds=3, seed=0, out=None)
    res = mb_behaviour.run(a)
    for row in res["tmaze_PI_by_beta"].values():
        assert "innate_bias_A_vs_B" in row and "PI_untrained" not in row
        assert row["PI_trained"] > 0                                           # punishing an approach-output odour -> avoidance
    over = res["tmaze_PI_over_seeds"]
    assert over["n_seeds"] == 3 and over["learned_score_shift_mean_sd"][0] > 0
    assert set(over["by_beta"]) == {4.0, 8.0} and over["by_beta"][8.0]["PI_mean_sd"][0] > 0
    b8 = over["by_beta"][8.0]                                                  # the seeds really draw different pairs
    assert b8["PI_mean_sd"][1] > 0 and b8["PI_min_max"][0] != b8["PI_min_max"][1] and over["learned_score_shift_mean_sd"][1] > 0
    assert res["punish_compartment_types"] == ["MBON11"] and res["reward_compartment_types"] == ["MBON03", "MBON05"]
    # the per-pair loop must start every pair from the UNTRAINED circuit: with one seed it must reproduce the example pair
    a.pi_seeds = 1; r1 = mb_behaviour.run(a)
    for beta, row in r1["tmaze_PI_by_beta"].items():
        assert r1["tmaze_PI_over_seeds"]["by_beta"][beta]["PI_mean_sd"][0] == row["PI_trained"]
        assert r1["tmaze_PI_over_seeds"]["by_beta"][beta]["abs_innate_bias_mean_sd"][0] == abs(row["innate_bias_A_vs_B"])
    ex = r1["learned_score_shift_example_pair"]
    assert abs(r1["tmaze_PI_over_seeds"]["learned_score_shift_mean_sd"][0] - 0.5 * (ex["A"] + ex["B"])) < 1e-4
    a.pi_seeds = 3; a.pairings = 0; res0 = mb_behaviour.run(a)                 # no training: the reciprocal index is identically 0
    assert all(abs(row["PI_trained"]) < 1e-9 for row in res0["tmaze_PI_by_beta"].values())
    a.pairings = 1


def test_lesion_rows_carry_learned_shift_and_identity_flags(wiring, neurons):
    from flybrain.eval import mb_lesion
    a = argparse.Namespace(subgraph=None, neurons=neurons, wiring=wiring, pairings=1, target=0.9, beta=8.0, seed=0, out=None)
    rows = {r["lesion"]: r for r in mb_lesion.run(a)["lesions"]}
    assert rows["none"]["learned_shift_frac_of_intact"] == 1.0 and rows["none"]["independent_wiring_test"]
    assert rows["PPL101_silenced"]["learned_shift"] == 0.0 and not rows["PPL101_silenced"]["independent_wiring_test"]
    assert rows["APL_disinhibited"]["drive_loss_vs_intact"] is None            # a dense code raises drive; not a loss
    assert rows["KCg-m_silenced"]["applied"] and rows["KCg-m_silenced"]["independent_wiring_test"]
    assert all("learned_shift" in r for r in rows.values())


def test_online_closed_form_matches_the_timed_episode_and_backward_writes_nothing(wiring):
    from flybrain.eval.mb_online import closed_form_drop, da_pulse_train, train_episode
    mb = mb_of(wiring); code = mb.kc_code(mb.odour([0, 1, 2])); m11 = mb.type_mask("MBON11")
    kw = {"dt": 0.01, "odour_on": 0.0, "odour_off": 2.0, "da_width": 0.05, "tau_elig": 0.8, "tau_forget": 1e9}
    d11 = float(mb.delta_punish[m11].max()); before = mb.mbon_response(code)
    fwd = da_pulse_train(4, 0.2, 2.0, 0.05)
    mb.reset(); train_episode(mb, code, "punish", eta=5.0, da_starts=fwd, **kw)
    measured = 1 - float(mb.mbon_response(code)[m11].sum() / before[m11].sum())
    assert 0.3 < measured < 1.0
    assert abs(measured - closed_form_drop(5.0, d11, da_starts=fwd, **kw)) < 1e-4   # binary code: the drop is the scalar closed form
    back = da_pulse_train(4, -1.0, 2.0, 0.05, anchor="end")
    assert back.max() + 0.05 <= -1.0 + 1e-9                                    # the whole train ends 1 s before odour onset
    mb.reset(); train_episode(mb, code, "punish", eta=5.0, da_starts=back, **kw)
    assert torch.equal(mb.W, torch.ones(mb.n_edges))                          # nothing written: zero by construction


def test_two_hop_relays_exclude_mbons_and_dns():
    from flybrain.eval.mb_build import mbon_dn_paths
    mbon = np.array([1, 2]); dn = np.array([10, 11]); X = 20
    # 1->X->10 is a real interneuron relay (3*4); 1->2->10 relays through an MBON and 1->11->10 through a DN: both excluded
    w = pd.DataFrame({"body_pre": [1, X, 1, 2, 1, 11, 1], "body_post": [X, 10, 2, 10, 11, 10, 10],
                      "weight": [3, 4, 5, 6, 7, 8, 9]})
    W1, W2, inter = mbon_dn_paths(w, mbon, dn)
    assert inter.tolist() == [X]
    assert W1[0, 0] == 9 and W1[0, 1] == 6                                     # direct MBON -> DN kept
    assert W2[0, 0] == 12 and W2[0, 1] == 0


def test_loo_decoder_scores_a_separable_code_and_not_a_shuffled_one():
    from flybrain.eval.mb_sparse import loo_nearest_centroid
    O, R = 6, 5; y = torch.arange(O).repeat_interleave(R); g = torch.Generator().manual_seed(0)
    K = torch.repeat_interleave(torch.eye(O), R, dim=0) + 0.05 * torch.randn(O * R, O, generator=g)
    assert loo_nearest_centroid(K, y) == 1.0
    assert loo_nearest_centroid(K, y[torch.randperm(O * R, generator=g)]) < 0.5
    flat = torch.ones(O * R, 50) + 0.01 * torch.randn(O * R, 50, generator=g)   # every odour the same pattern: no code
    assert loo_nearest_centroid(flat, y) < 0.35                                # near chance; a resubstitution decoder would score ~1
    assert np.isnan(loo_nearest_centroid(K[::R], torch.arange(O)))            # one sample per class: undefined, not 0


def test_online_run_labels_trains_and_its_closed_form_holds_everywhere(wiring, neurons):
    """Run-level guard: a negative onset yields a train wholly before the odour (writes nothing), the anchor point
    reproduces the target, a train after odour offset writes a partial memory, and the printed closed form matches
    the measured drop at every timing and dose point (binary code: the curves are the rule, not the wiring)."""
    from flybrain.eval import mb_online
    a = argparse.Namespace(subgraph=None, neurons=neurons, wiring=wiring, dt=0.01, odour_dur=2.0, pulses=4, da_onset=0.2,
                           da_freq=2.0, da_width=0.05, tau_elig=0.8, tau_forget=1e9, target=0.9,
                           timing_onsets=[-1.0, 0.2, 2.2], dose_pulses=[0, 2, 4], beta=8.0, seed=0, out=None)
    res = mb_online.run(a); t = res["timing_MBON11"]
    assert t["-1.0"]["relation_to_odour"] == "before odour" and t["-1.0"]["pulses_in_odour"] == 0 and t["-1.0"]["drop_MBON11"] == 0.0
    assert t["-1.0"]["train_s"][1] <= -1.0 + 1e-9
    assert t["0.2"]["relation_to_odour"] == "during odour" and abs(t["0.2"]["drop_MBON11"] - 0.9) < 1e-3      # the anchor
    assert t["2.2"]["relation_to_odour"] == "after odour" and 0.0 < t["2.2"]["drop_MBON11"] < 0.9
    for row in list(t.values()) + list(res["training_amount_MBON11"].values()):
        assert abs(row["drop_MBON11"] - row["closed_form_MBON11"]) < 2e-4
    assert res["mechanism_controls"]["dopamine_alone_no_odour"] == 0.0 and res["mechanism_controls"]["odour_alone_no_dopamine"] == 0.0
    assert res["behaviour_from_timed_teaching"]["T_maze_PI"]["backward_taught"] == 0.0


def test_seeds_aggregate_spans_different_draws(wiring):
    from flybrain.eval import mb_seeds
    a = argparse.Namespace(wiring=wiring, modalities=["olfactory"], seeds=3, seed=0, out_dir=None)
    s = mb_seeds.run(a)["per_modality"]["olfactory"]
    assert s["paired_A_MBON11"]["n"] == 3 and abs(s["paired_A_MBON11"]["mean"] - 0.9) < 1e-6 and s["paired_A_MBON11"]["sd"] == 0.0
    assert s["unpaired_MBON11_mean"]["sd"] > 0 and s["unpaired_MBON11_worst_odour"]["max"] >= s["unpaired_MBON11_mean"]["max"]
    assert set(s["generalisation_drop_MBON11"]) == {"shared_5_of_6", "shared_4_of_6", "shared_3_of_6", "shared_1_of_6", "shared_0_of_6"}
    assert s["circuit"]["n_drivable_KC"] == N_DRIVABLE


def test_apl_loop_sparsens_with_gain(wiring):
    from flybrain.eval.mb_apl import apl_code
    z = np.load(wiring); mb = mb_of(wiring)
    drive = np.asarray(mb.W_pk @ mb.odour([0, 1, 2])).ravel()
    fracs = [float((apl_code(drive, z["kc_apl"].astype(float), z["apl_kc"].astype(float), g, 500) > 0).mean()) for g in (1, 10, 100)]
    assert fracs[0] >= fracs[1] >= fracs[2] and fracs[2] < fracs[0]
