"""Precompute the mushroom-body wiring submatrices once, from the FULL MaleCNS connectome (minconf 0.5, no
weight threshold), so the olfactory and visual memory circuits sit on the same faithful substrate rather than
the weight-thresholded Doom subgraph (which prunes 57 percent of the weak visual KC input). Writes a small
npz the MushroomBody loads in a fraction of a second.

  python -m flybrain.eval.mb_build --data $FLYBRAIN_DATA/malecns_v1 --out $FLYBRAIN_OUT/mb/mb_wiring.npz
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flybrain import DATA_DIR, OUT_DIR

PN_SUF = ("adPN", "lPN", "ilPN", "l2PN", "il2PN")

def is_olfactory_pn(t: str) -> bool:
    if "_" not in t: return False
    g, s = t.rsplit("_", 1)
    return s in PN_SUF and "+" not in g and not g.startswith("VP")


def mbon_dn_paths(w: pd.DataFrame, mbon: np.ndarray, dn: np.ndarray):
    """MBON -> DN direct and MBON -> one interneuron -> DN two-hop capacity, both [DN, MBON], from a weights table
    with columns body_pre, body_post, weight. A relay is an INTERNEURON: cells that are themselves MBONs or DNs are
    excluded, so the two-hop matrix never re-counts the direct path or an MBON -> MBON -> DN chain. Returns
    (direct, two_hop, relay_bodyIds)."""
    mo = w[w.body_pre.isin(set(mbon))]                           # MBON outputs
    di = w[w.body_post.isin(set(dn))]                            # DN inputs
    inter = np.array(sorted(set(mo.body_post) & set(di.body_pre)))   # neurons MBON drives that also drive DNs
    inter = inter[~np.isin(inter, np.concatenate([mbon, dn]))]        # a relay is an interneuron: not itself an MBON or a DN
    def mat(df, rows, cols):
        ri = {b: i for i, b in enumerate(rows)}; ci = {b: i for i, b in enumerate(cols)}
        s = df[df.body_post.isin(set(rows)) & df.body_pre.isin(set(cols))]
        return sparse.csr_matrix((s.weight.to_numpy(float), ([ri[b] for b in s.body_post], [ci[b] for b in s.body_pre])), shape=(len(rows), len(cols)))
    W1 = np.asarray(mat(mo[mo.body_post.isin(set(dn))], dn, mbon).todense())      # [DN, MBON] direct
    Mi = mat(mo, inter, mbon)                                    # [inter, MBON]
    Id = mat(di, dn, inter)                                      # [DN, inter]
    W2 = np.asarray((Id @ Mi).todense())                        # [DN, MBON] 2-hop
    return W1, W2, inter


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR / "malecns_v1")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    a = ap.parse_args(argv); t0 = time.time()
    ann = pd.read_feather(a.data / "body-annotations-male-cns-v1.0-minconf-0.5.feather")[["bodyId", "superclass", "type"]]
    ann["type"] = ann["type"].fillna("").astype(str); ann["superclass"] = ann["superclass"].fillna("").astype(str)
    w = pd.read_feather(a.data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

    kc = ann.bodyId[ann["type"].str.startswith("KC")].to_numpy()
    kc_type = ann.set_index("bodyId")["type"].reindex(kc).to_numpy()
    mbon = ann.bodyId[ann["type"].str.startswith("MBON")].to_numpy()
    mbon_type = ann.set_index("bodyId")["type"].reindex(mbon).to_numpy()
    opn = ann.bodyId[ann["type"].map(is_olfactory_pn)].to_numpy()
    opn_glom = np.array([str(t).split("_")[0] for t in ann.set_index("bodyId")["type"].reindex(opn)])
    # visual PNs that actually reach KCs; label each by its VPN type (feature channel, e.g. LC / LPLC)
    vpn_all = ann.bodyId[ann.superclass == "visual_projection"].to_numpy()
    ppl1 = ann.bodyId[ann["type"].str.startswith("PPL1")].to_numpy()
    pam = ann.bodyId[ann["type"].str.startswith("PAM")].to_numpy()
    dan = np.concatenate([ppl1, pam]); dan_type = ann.set_index("bodyId")["type"].reindex(dan).to_numpy()

    def submat(pre, post):
        s = w[w.body_post.isin(set(post)) & w.body_pre.isin(set(pre))]
        pi = {b: i for i, b in enumerate(pre)}; qi = {b: i for i, b in enumerate(post)}
        return sparse.csr_matrix((s.weight.to_numpy(float), ([qi[b] for b in s.body_post], [pi[b] for b in s.body_pre])),
                                 shape=(len(post), len(pre)))

    W_pk = submat(opn, kc)                                        # [KC, PN]
    W_vk_all = submat(vpn_all, kc)                               # [KC, all VPN]
    conn = np.flatnonzero(np.asarray((W_vk_all > 0).sum(0)).ravel() > 0)   # VPNs that reach any KC
    vpn = vpn_all[conn]; W_vk = W_vk_all[:, conn].tocsr()
    vpn_type = ann.set_index("bodyId")["type"].reindex(vpn).fillna("").to_numpy()
    W_km = submat(kc, mbon).tocoo()                              # [MBON, KC]
    W_dan = np.asarray(submat(dan, mbon).todense())             # [MBON, nDAN] dense (small)
    # total synaptic input onto each MBON from the WHOLE connectome, so "fraction of MBON11's input that comes
    # from Kenyon cells" is a recorded wiring number, not a quoted one
    mbon_in_total = w[w.body_post.isin(set(mbon))].groupby("body_post").weight.sum().reindex(mbon).fillna(0.0).to_numpy(float)

    # APL feedback inhibitor: KC -> APL (drives it) and APL -> KC (inhibits them). The real substrate for sparse
    # coding, so the Kenyon code can be computed from the wiring instead of a hand-set k-winners-take-all.
    apl = ann.bodyId[ann["type"].str.startswith("APL")].to_numpy()
    W_kc_apl = np.asarray(submat(kc, apl).todense()) if len(apl) else np.zeros((0, len(kc)))   # [APL, KC]
    W_apl_kc = np.asarray(submat(apl, kc).todense()) if len(apl) else np.zeros((len(kc), 0))   # [KC, APL]

    # memory -> steering: MBON -> DN direct, and MBON -> one interneuron -> DN (2-hop), both [DN, MBON]
    dn = ann.bodyId[ann.superclass == "descending_neuron"].to_numpy()
    dn_type = ann.set_index("bodyId")["type"].reindex(dn).fillna("").to_numpy()
    W1, W2, inter = mbon_dn_paths(w, mbon, dn)

    # string dtypes must not truncate a type name (a truncated 'KCg-m' or 'DNa02' would silently mis-select cells)
    for name, arr, width in (("kc_type", kc_type, 16), ("mbon_type", mbon_type, 16), ("dan_type", dan_type, 16),
                             ("dn_type", dn_type, 16), ("vpn_type", vpn_type, 24), ("opn_glom", opn_glom, 16)):
        longest = max((len(str(x)) for x in arr), default=0)
        if longest > width:
            raise ValueError(f"{name}: a type name is {longest} chars, wider than the U{width} cache dtype; widen it")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    src_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    np.savez_compressed(a.out,
        kc_type=kc_type.astype("U16"), mbon_type=mbon_type.astype("U16"),
        opn_glom=opn_glom.astype("U16"), vpn_type=vpn_type.astype("U24"), dan_type=dan_type.astype("U16"),
        pk_data=W_pk.data, pk_indices=W_pk.indices, pk_indptr=W_pk.indptr, pk_shape=W_pk.shape,
        vk_data=W_vk.data, vk_indices=W_vk.indices, vk_indptr=W_vk.indptr, vk_shape=W_vk.shape,
        km_row=W_km.row, km_col=W_km.col, km_data=W_km.data, km_shape=W_km.shape,
        dan_mbon=W_dan, dn_type=dn_type.astype("U16"), dn_mbon_direct=W1, dn_mbon_2hop=W2,
        kc_apl=W_kc_apl, apl_kc=W_apl_kc,
        mbon_in_total=mbon_in_total, kc_body=kc.astype(np.int64),          # provenance for readouts and the video
        build_time=np.array(time.strftime("%Y-%m-%dT%H:%M:%S")), build_source_sha=np.array(src_sha))
    vt = pd.Series(vpn_type); gt = pd.Series(opn_glom)
    drv_o = np.asarray((W_pk != 0).sum(1)).ravel() > 0; drv_v = np.asarray((W_vk != 0).sum(1)).ravel() > 0
    print(f"KC {len(kc)} | olf PN {len(opn)} ({gt.nunique()} glom) | visual PN {len(vpn)} ({vt.nunique()} types) | "
          f"MBON {len(mbon)} | DAN {len(dan)} | W_pk {W_pk.nnz} W_vk {W_vk.nnz} W_km {W_km.nnz}")
    print(f"KCs drivable by olfactory PNs {int(drv_o.sum())} | by visual PNs {int(drv_v.sum())} (the visual kWTA pool), by KC type: "
          f"{dict(pd.Series(kc_type[drv_v]).value_counts())}")
    print(f"DN {len(dn)} | MBON->DN direct nz {int((W1>0).sum())} | MBON->X->DN 2-hop via {len(inter)} interneurons (MBONs and DNs excluded as relays), nz {int((W2>0).sum())}")
    print("visual feature channels (VPN types reaching KCs), top 15:", dict(vt.value_counts().head(15)))
    print(f"wrote {a.out} (build_source_sha {src_sha}) in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
