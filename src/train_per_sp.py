"""PatchFormer-consistent per-SP protocol training (10 seeds per SP).

Protocol (user decision 2026-08-17, STRICT PatchFormer consistency):

  For every dataset, for EVERY (SP, seed) pair:
    - train a fresh model (10 seeds per SP: 1..10)
    - train = non-test cells' FULL sequences
             PLUS the test cell's cycles BEFORE the SP (Cycle < SP)
             --- exactly the NASADataPreProcess.py convention, applied
                uniformly to all five datasets per user instruction ---
    - per-window z-score target; global min-max input from train split
    - evaluate the segment starting at SP (absolute index), with
      per-window de-normalization; TRUL/PRUL/AE/MAE/RMSE/R2

Outputs:
  checkpoints/per_sp/{ds}/SP{sp}_seed{seed}.pt
  results/per_sp_train.json  (per dataset/SP/seed metrics + 10-seed means)
"""
import argparse
import json
import time
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series
from eval_multiseed import true_rul

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH is env-overridable because this GPU is idle at 64 samples: a
# forward+backward at batch 512 costs the same 1.1 s as one at batch 64
# (measured), so a rollout that is sequential in TIME but can be batched over
# SAMPLES is 8x cheaper at 512 than at 64.  Default unchanged.
BATCH = int(os.environ.get("BATCH", "64"))
# EPOCHS is env-overridable because a scheduled-sampling epoch is not one
# epoch's worth of learning: each sample yields K gradient signals instead of
# one, so a shorter run is the equivalent of a much longer plain one.  The
# default is unchanged, so every existing invocation trains exactly as before.
EPOCHS = int(os.environ.get("EPOCHS", "100"))
EPS = 1e-6

# ABS_TARGET=1 switches the training target from the window-relative
# z-score to the normalised capacity itself (see the note at the loss).
#
# Everything an absolute-target run touches is namespaced so the z-score runs
# stay byte-identical: checkpoints get an `_abs` suffix and the metrics go to
# results/per_sp_train_abs.json instead of results/per_sp_train.json.
#
# CAVEAT: eval_sp() decodes with the z-score rule (pred * wstd + wmean), which
# is WRONG for an absolute-target model -- its rows are not ABS metrics.  The
# correct decode is y_hat = model(x) (see src/ar_abs_rollout.py and
# docs/ar_freeze_findings.md section 7); recompute before quoting anything.
import os as _os
ABS_TARGET = _os.environ.get("ABS_TARGET", "") == "1"

# TGT_MODE picks the training target AND the matching decode.  The three modes
# differ in two independent ways, which are worth keeping straight because they
# fail differently:
#
#   zscore   tgt = (y - wmean)/wstd   dec = z*wstd + wmean
#            level anchored to the window mean, output scaled by wstd.
#            Stable, but wstd collapses under autoregression -> the rollout
#            freezes at the window mean (docs/ar_freeze_findings.md).
#   abs      tgt = y                  dec = pred
#            DIRECT absolute regression: the head must learn the level itself.
#            It underfits by shrinking toward the mean (a = 0.96-0.99 measured on
#            both CALCE and PoliMi, losing to a linear ridge fit on the same
#            windows), and that compression compounds over a rollout.
#   anchor   tgt = y - wmean          dec = pred + wmean
#            level anchored to the window mean (no shrink) but NOT scaled by
#            wstd (no freeze).  The target is zero-centred with a range of a few
#            hundredths -- the same scale the head starts at -- so it is far
#            better conditioned than `abs`.
#
# TGT_MODE defaults to whatever ABS_TARGET implies, so every existing run keeps
# its behaviour byte-for-byte.
TGT_MODE = _os.environ.get("TGT_MODE", "abs" if ABS_TARGET else "zscore")
# anchor_last is the 2x2 cell the other three leave empty:
#
#                 anchor on mu_win      anchor on Q_last    no anchor
#   no sigma      anchor  (fails)       anchor_last (?)     abs  (runs)
#   with sigma    zscore  (fails)       --                  --
#
# zscore and anchor BOTH subtract the window MEAN, and both die under rollout;
# abs subtracts nothing and runs.  That correlation is too clean to be a
# coincidence, so anchor_last isolates it: same no-sigma decode as anchor, but
# anchored on the last observed value instead of the window mean.  It is also
# exactly the phys_ir rate head with the softplus removed, so it answers "is
# monotonicity load-bearing?" at the same time.
assert TGT_MODE in ("zscore", "abs", "anchor", "anchor_last"), TGT_MODE

# READOUT selects the output head.  "last" is the paper's main model and the
# default, so every existing run is bit-identical.
#
#   last     head_cap(h)         free head; pairs with the z-score/anchor target
#   phys_ir  Q_last - softplus(h)  rate head: monotone by construction and
#                                 anchored on the last OBSERVED value, which is
#                                 what makes it the only head here that can in
#                                 principle be driven forward on its own
#                                 outputs.  Train it with TGT_MODE=abs: its raw
#                                 output already IS the next absolute capacity.
#
# The rate head is the candidate for the paper's rollout claim -- z-score and
# anchor both decode through the window MEAN, and that decode provably has a
# repelling fixed point (z_crit), so neither can ever forecast forward.  See
# docs/ar_freeze_findings.md.
READOUT = _os.environ.get("READOUT", "last")
assert READOUT in ("last", "mean", "phys_ir", "phys"), READOUT

# AR_LOSS=1 adds a K-step differentiable autoregressive rollout to the loss.
# The model has never been trained on the windows autoregression actually
# feeds it (the evaluation region's real windows bottom out at sigma = 0.0081;
# a collapsed rollout reaches 0.00024), so this is the one lever left once the
# target and the decode are both fixed.  Rationale and the alternatives that
# were measured and rejected are in docs/ar_freeze_findings.md.
#
# The loss is in CAPACITY space, not z space: the two differ only by the
# per-step weight sigma_k, and z space weights by 1/sigma_k -- so it would put
# almost all the weight on the last steps, whose window no longer carries
# information about the true state.
AR_LOSS = _os.environ.get("AR_LOSS", "") == "1"
AR_EVERY = int(_os.environ.get("AR_EVERY", "8"))      # apply to 1 in N batches
# checkpoint/json tag, so a re-run with a changed implementation does not
# overwrite the previous attempt's artefacts
AR_TAG = _os.environ.get("AR_TAG", "ar")
AR_WARM = int(_os.environ.get("AR_WARM", "10"))       # epochs of pure TF first
# lambda is NOT a free parameter.  The TF term is a z-space error (~0.42) and
# the AR term a capacity-space error (~0.011 at epoch 10), and the two differ
# by ~1/sigma -- measured at 37x.  A fixed lambda therefore silently turns into
# "no AR term at all" (lambda=0.5 puts it at 1.3% of the loss).  Derive it from
# the observed ratio instead, so the AR term holds AR_FRAC of the total.
AR_FRAC = float(_os.environ.get("AR_FRAC", "0.25"))
AR_LAM_MAX = float(_os.environ.get("AR_LAM_MAX", "50"))
# print one progress line every PROG_EVERY epochs (epoch index, mean loss,
# elapsed, ETA); 0 silences it.  Training is long enough that silence is
# indistinguishable from a hang.
PROG_EVERY = int(_os.environ.get("PROG_EVERY", "5"))
# PROG_SAVE > 0 keeps a checkpoint every PROG_SAVE epochs, so a long run can be
# evaluated part-way instead of only at the end.  train_one writes these only
# when the caller passes save_prefix, so every existing caller is unaffected.
PROG_SAVE = int(_os.environ.get("PROG_SAVE", "25"))
# (epoch below, K).  Cost per epoch is linear in K, so the curriculum is the
# cost driver -- K = 32 in the last 30 epochs roughly quadrupled the run.  K is
# capped because the measured z crosses the self-consistency threshold around
# rollout step 40-80, so longer rollouts buy little and cost linearly.
#
# AR_KS is the K of each stage, spread evenly over EPOCHS: AR_KS=4,8,16 with
# EPOCHS=100 gives K=4 for epochs 0-32, 8 for 33-65, 16 after.  AR_KMAX is the
# largest of them and is how many future values build_windows keeps.
AR_KS = tuple(int(v) for v in _os.environ.get("AR_KS", "4,8,16").split(",")
              if v.strip())
AR_KMAX = max(AR_KS)
_AR_STAGE = max(1, EPOCHS // len(AR_KS))
AR_SCHEDULE = tuple(
    ((i + 1) * _AR_STAGE if i < len(AR_KS) - 1 else 10 ** 9, k)
    for i, k in enumerate(AR_KS))


def ar_weight(ep: int) -> tuple[int, float]:
    """(K, ramp).  Both ramp: an AR loss from epoch 0 fights the one-step fit
    before there is anything to roll out.  The returned ramp is a fraction in
    [0,1]; the caller turns it into a lambda once it has seen both losses."""
    K = next(k for lim, k in AR_SCHEDULE if ep < lim)
    if ep < AR_WARM:
        return K, 0.0
    return K, min(1.0, (ep - AR_WARM + 1) / 20.0)


# SCHED_SAMPLE = p turns on scheduled sampling (Bengio et al. 2015): along a
# K-step rollout of the batch's own windows, each step feeds back the model's
# own previous prediction with probability p and the TRUE value otherwise.
# p=0 leaves training exactly as it was.
#
# This is NOT AR_LOSS, and the three differences are the whole point:
#   * AR_LOSS feeds back predictions unconditionally; here it is a per-step
#     coin flip, so the model still sees clean windows half the time.
#   * AR_LOSS backprops through the fed-back value; here it is detached, so
#     each step's gradient comes only from that step's own forward.
#   * AR_LOSS is an EXTRA loss term fighting the teacher-forced term (and it
#     needed an adaptively-derived lambda to stay visible at all).  Here the
#     SS loss IS the loss.
#
# Motivated by Yao, Zhao & Kowal, J. Power Sources 660 (2025) 238569, who get
# an encoder-decoder to roll out recursively on CALCE (MAE 0.603-1.156%) using
# exactly this technique at p=0.5 while their baseline model fails outright
# (5.993%).  See docs/ar_freeze_findings.md.
SCHED_SAMPLE = float(_os.environ.get("SCHED_SAMPLE", "0"))
# K is capped by AR_KMAX because XK/KM only carry AR_KMAX future values.
SCHED_K = min(int(_os.environ.get("SCHED_K", "8")), AR_KMAX)
# Backward each rollout step as it is produced instead of holding all K graphs
# to the end.  See the note in sched_sample_loss: at BATCH=512, K=8 this is
# what keeps the run inside an 8 GB card.
SCHED_STEP_BACKWARD = _os.environ.get("SCHED_STEP_BACKWARD", "1") == "1"


def _mode_suffix(early_stop: bool) -> str:
    """Checkpoint suffix for the active mode; "" is the plain z-score run.

    A non-default READOUT gets its own prefix.  A rate-head run and a free-head
    run share a dataset, an SP and a seed but are different models, so without
    the prefix the second one would silently overwrite the first -- and the
    only thing worse than losing the checkpoint is scoring the wrong one.
    """
    head = "" if READOUT == "last" else (
        "rate" if READOUT == "phys_ir" else READOUT)
    # A non-default batch is a different training run, not a detail of one: a
    # batch-512 control would otherwise produce the SAME suffix as the paper's
    # batch-64 model ("") and overwrite SP{sp}_seed{n}.pt with it.
    if BATCH != 64:
        head = f"{head}b{BATCH}" if head else f"b{BATCH}"
    if TGT_MODE != "zscore":
        tag = {"abs": "abs", "anchor": "anchor",
               "anchor_last": "alast"}[TGT_MODE]
        tag = f"{head}_{tag}" if head else tag
        return f"_es_{tag}" if early_stop else f"_{tag}"
    if AR_LOSS:
        tag = f"{head}_{AR_TAG}" if head else AR_TAG
        return f"_es_{tag}" if early_stop else f"_{tag}"
    if SCHED_SAMPLE > 0:
        # K is part of the identity of an SS run: K=32 and K=8 are different
        # models trained on different input distributions, so they must not
        # share a checkpoint name -- two concurrent runs would otherwise
        # interleave writes into the same file.
        ss = "ss" if SCHED_K == 8 else f"ssk{SCHED_K}"
        tag = f"{head}_{ss}" if head else ss
        return f"_es_{tag}" if early_stop else f"_{tag}"
    if head:
        return f"_es_{head}" if early_stop else f"_{head}"
    return "_es" if early_stop else ""


def build_windows(caps, cells, lo, hi, W, max_cycle=None, with_future=False,
                  kmax=0):
    """Windows from the given cells' sequences (or up to max_cycle).

    max_cycle = SP: includes ONLY cycles < SP (PatchFormer convention).

    with_future=True additionally returns (XK, KM): the next `kmax` true values
    after each window's target, and a 1/0 mask for the ones that exist.  The
    continuation is capped at `end` -- the same limit the windows obey -- so a
    test-cell window's future never reaches a held-out cycle.

    with_future=False returns (X, Y), which is what the other three callers
    unpack; the third and fourth values are opt-in for exactly that reason.
    """
    X, Y, XK, KM = [], [], [], []
    for c in cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        end = len(seq) if max_cycle is None else min(len(seq), max_cycle)
        for i in range(W, end):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
            if with_future:
                fut = seq[i:min(i + kmax, end)]      # never past `end`
                XK.append(np.pad(fut, (0, kmax - len(fut))))
                KM.append(np.pad(np.ones(len(fut), np.float32),
                                 (0, kmax - len(fut))))
    if not with_future:
        return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32)
    return (np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32),
            np.stack(XK).astype(np.float32),
            np.stack(KM).astype(np.float32))


def ar_rollout_loss(model, win_true, xk, km, K):
    """K-step differentiable autoregressive rollout, in capacity space.

    Every window in the training batch is rolled out -- no separate sampling.
    The batch is already a uniform draw over the training split, and K steps
    cost K model calls whether the batch is 4 or 64, so subsampling bought
    nothing but variance.

      win_true : (B, W)   the batch's true windows
      xk       : (B, KMAX) the true values the window at each step predicts
      km       : (B, KMAX) 1 where that step exists, 0 where the series ended
      K        : how many steps to roll out (must be <= KMAX)

    Step 0 predicts from a true window, so it restates the teacher-forcing
    term in a different space; steps 1..K-1 are the autoregressive part, where
    the window holds the model's own output and the gradient flows back along
    that chain.  The append is torch.cat rather than an in-place assignment:
    both keep the graph (checked), but in-place relies on that tensor not being
    saved for backward.

    |y_hat - y| == sigma_k * |z - (y - mu_k)/sigma_k|, so capacity space is z
    space weighted by sigma_k.  z space weights by 1/sigma_k, and sigma
    collapses over a rollout, so it would put the weight on the steps whose
    window no longer carries information.  mu and sigma are detached: attached,
    the model can lower the loss by reshaping the window statistics, and
    "flatten the window until the prediction sits on the mean" is exactly the
    degenerate solution that freezes the rollout.
    """
    win = win_true.clone()
    # a tensor, not a Python float: if the loop breaks at k = 0 the caller
    # still gets a tensor back, and `ar.item()` in train_one does not explode
    total, n = torch.zeros((), device=win.device), 0
    for k in range(K):
        valid = km[:, k]
        nv = valid.sum()
        if float(nv) == 0.0:
            break
        mu = win.mean(dim=1)
        sig = win.std(dim=1) + EPS
        z = model(win.unsqueeze(-1)).squeeze(-1)
        y_hat = z * sig.detach() + mu.detach()
        total = total + ((y_hat - xk[:, k]).abs() * valid).sum() / nv
        win = torch.cat([win[:, 1:], y_hat.unsqueeze(1)], dim=1)
        n += 1
    # n == 0 leaves `total` a tensor of zeros, so the return type is stable
    return total / max(n, 1)


def sched_sample_loss(model, win_true, xk, km, K, p):
    """Scheduled sampling over a REAL autoregressive rollout of K steps.

    At each step the window entry past the observed history is the model's own
    prediction with probability p and the true value otherwise, so the model
    trains on the input distribution it will meet at inference (where p is
    effectively 1) without ever being trained on nothing else.

    The rollout is genuinely autoregressive: the prediction at step k is made
    from a window that already contains earlier predictions, so contamination
    feeds back -- which is what Yao et al.'s decoder does and what makes the
    technique more than input noise.  That makes the K steps sequential in
    time, and the first version of this function was written serial AND at the
    training batch size, which cost 8.7x a plain epoch for no reason: this GPU
    is nowhere near saturated at 64 samples, so a forward+backward at batch 512
    costs the same 1.1 s as one at batch 64 (measured).  The rollout is
    sequential in TIME but batchable over SAMPLES, so the fix is a bigger
    batch, not a cleverer loop -- at BATCH=512 the same epoch costs 50 s
    instead of 412 s.

    Fed-back values are DETACHED: this changes the INPUT distribution, it does
    not add a second objective, and letting gradient flow back through the
    rollout would make the gradient grow with K.  Because each fed-back value
    is detached, the K steps' graphs are independent, so the memory held until
    backward is K separate graphs rather than one deep chain.
    """
    B, W = win_true.shape
    # XK/KM are built with AR_KMAX columns because AR_LOSS stages its K up to
    # that; scheduled sampling may want fewer.  Slice rather than assume.
    xk, km = xk[:, :K], km[:, :K]
    win = win_true.clone()
    total = torch.zeros((), device=win.device)
    for k in range(K):
        valid = km[:, k]
        nv = valid.sum().clamp(min=1.0)
        mu = win.mean(dim=1)
        sig = win.std(dim=1) + EPS
        raw = model(win[:, :, None]).reshape(-1)
        y = decode_pred(raw, mu, sig)          # capacity space, any TGT_MODE
        loss_k = ((y - xk[:, k]).abs() * valid).sum() / nv
        if SCHED_STEP_BACKWARD:
            # The fed-back value is detached, so each step's graph is already
            # independent; backwarding here frees it instead of holding all K.
            # At BATCH=512, K=8 that is the difference between 7.9 GB and a
            # 8.2 GB card -- i.e. between training and thrashing.
            (loss_k / K).backward()
            total = total + loss_k.detach()
        else:
            total = total + loss_k
        use_pred = torch.rand_like(xk[:, k]) < p
        nxt = torch.where(use_pred, y.detach(), xk[:, k])
        win = torch.cat([win[:, 1:], nxt[:, None]], dim=1)
    out = total / K
    # Detached when the steps already backpropped themselves, so the caller's
    # own loss.backward() becomes a no-op rather than double-counting.
    return out.detach() if SCHED_STEP_BACKWARD else out


def train_one(seed, X, Y, W, XK=None, KM=None, save_prefix=None,
              save_meta=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout=READOUT,
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    step = 0
    lam_fixed = None          # calibrated once, then frozen; see below
    t_wall = time.time()
    last_t, last_ep = t_wall, 0
    for ep in range(EPOCHS):
        model.train()
        ar_on = AR_LOSS and XK is not None
        ss_on = SCHED_SAMPLE > 0 and XK is not None
        K, ramp = ar_weight(ep) if ar_on else (0, 0.0)
        perm = np.random.permutation(N)
        ep_loss, nb, ep_tf, ep_ar, nb_ar, lam = 0.0, 0, 0.0, 0.0, 0, 0.0
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            # ABS_TARGET=1 trains on the normalised capacity itself instead of
            # the window-relative z-score.  The z-score target is what forces
            # the decode `y = z * std(window) + mean(window)` at inference, and
            # that decode freezes under autoregressive rollout: the window
            # fills with the model's own (smooth) output, its std collapses,
            # and the prediction sticks at the window mean.
            tgt = target_of(y, wmean, wstd, ylast=x[:, -1, 0])
            if ss_on:
                # Scheduled sampling IS the loss here.  Its first step sees the
                # untouched true window, so ordinary teacher-forced accuracy is
                # still trained; the later steps are where the model meets its
                # own output.
                loss = sched_sample_loss(
                    model, x[:, :, 0],
                    torch.tensor(XK[idx]).to(DEV),
                    torch.tensor(KM[idx]).to(DEV), SCHED_K, SCHED_SAMPLE)
            else:
                loss = masked_mae(pred, tgt, torch.ones_like(y))
            if ramp > 0.0 and step % AR_EVERY == 0:
                # roll out THIS batch's own windows -- same data as the TF term
                ar = ar_rollout_loss(
                    model, x[:, :, 0],
                    torch.tensor(XK[idx]).to(DEV),
                    torch.tensor(KM[idx]).to(DEV), K)
                if lam_fixed is None:
                    # lambda is a units conversion (z-space TF vs capacity-space
                    # AR) as much as a weight, so it comes from the first
                    # observed ratio -- and is then FROZEN.  Letting it track
                    # the ratio every step is a positive feedback loop: tf
                    # degrades -> lambda grows -> the AR term pushes harder ->
                    # tf degrades further.  That loop is what destabilised the
                    # first attempt (epoch-mean loss 0.31 <-> 0.72 over
                    # epochs 71-91).
                    lam_fixed = AR_FRAC * loss.item() / (ar.item() + 1e-9)
                    print(f"    [ar] calibrated lam={lam_fixed:.2f} "
                          f"from tf={loss.item():.5f} ar={ar.item():.5f}",
                          flush=True)
                lam = ramp * min(lam_fixed, AR_LAM_MAX)
                loss = loss + lam * ar
                ep_tf += float(loss.detach()) - float((lam * ar).detach())
                ep_ar += float(ar.detach())
                nb_ar += 1
            # When the SS steps backprop themselves the returned tensor is
            # detached, and backward() on a tensor without a grad_fn RAISES
            # rather than being a no-op -- so the call has to be skipped, not
            # left to sort itself out.
            if not (ss_on and SCHED_STEP_BACKWARD):
                loss.backward()
            opt.step()
            step += 1
            ep_loss += float(loss.detach())
            nb += 1
        ar_note = (f"  | tf={ep_tf / nb_ar:.5f} ar={ep_ar / nb_ar:.5f}"
                   f" lam={lam:.2f} K={K}" if nb_ar else "")
        if PROG_EVERY and (ep % PROG_EVERY == 0 or ep == EPOCHS - 1):
            now = time.time()
            # cost per epoch is NOT stationary (it is linear in AR K), so the
            # ETA comes from the epochs since the last print, not the run
            # average -- the average under-reports badly once K steps up
            per_ep = (now - last_t) / max(1, ep + 1 - last_ep)
            eta = per_ep * (EPOCHS - ep - 1)
            print(f"    [ep {ep + 1:>3}/{EPOCHS}] loss={ep_loss / nb:.5f}"
                  f"  {now - t_wall:.0f}s, {per_ep:.1f}s/ep, eta {eta:.0f}s"
                  f"{ar_note}", flush=True)
            last_t, last_ep = now, ep + 1
        # keep an intermediate checkpoint so a long run can be evaluated
        # part-way; written only when the caller asked for it
        if save_prefix and PROG_SAVE and (ep + 1) % PROG_SAVE == 0:
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "W": W, "ep": ep + 1, **(save_meta or {})},
                       f"{save_prefix}_ep{ep + 1:03d}.pt")
    return model


def train_one_earlystop(seed, X, Y, W, val_frac=0.2, patience=10,
                        min_delta=1e-5, max_epochs=200):
    """train_one + official-style validation.

    The validation split is the official row-order tail (last
    val_frac of the concatenated train frame, as in the PatchFormer/
    RUL-Mamba trainers); val loss uses the same per-window z-score
    target definition.  Best checkpoint (lowest val loss) is kept and
    training stops after `patience` epochs without improvement.
    Returns (best_model, best_epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout=READOUT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    n_tr = int((1 - val_frac) * N)
    Xt, Yt, Xv, Yv = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]

    def val_loss():
        model.eval()
        with torch.no_grad():
            x = torch.tensor(Xv, device=DEV)
            y = torch.tensor(Yv, device=DEV)
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            return float(masked_mae(pred, target_of(y, wmean, wstd,
                                                    ylast=x[:, -1, 0]),
                                    torch.ones_like(y)))

    best_loss, best_epoch, best_state = float("inf"), 0, None
    bad = 0
    for ep in range(max_epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(Xt[idx], device=DEV)
            y = torch.tensor(Yt[idx], device=DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            loss = masked_mae(pred, target_of(y, wmean, wstd,
                                              ylast=x[:, -1, 0]),
                              torch.ones_like(y))
            loss.backward()
            opt.step()
        vl = val_loss()
        if PROG_EVERY and (ep % PROG_EVERY == 0 or ep == max_epochs - 1):
            print(f"    [ep {ep + 1:>3}/{max_epochs}] val={vl:.5f}"
                  f"  best={best_loss:.5f}@{best_epoch}"
                  f"  no-improve {bad}", flush=True)
        if vl < best_loss - min_delta:
            best_loss, best_epoch, bad = vl, ep + 1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def target_of(y, wmean, wstd, mode=None, ylast=None):
    """The one place the four training targets differ.

    `ylast` is x[:, -1, 0]: the last observed capacity in the window.  Only
    anchor_last uses it, and it must come from the caller because it is not
    recoverable from (y, wmean, wstd).
    """
    m = mode if mode is not None else TGT_MODE
    if m == "abs":
        return y
    if m == "anchor":
        return y - wmean
    if m == "anchor_last":
        assert ylast is not None, "anchor_last needs the window's last value"
        return y - ylast
    return (y - wmean) / wstd


def window_std(x):
    """The decode's window scale -- literally the operator training used.

    Training computes the target's scale as `x[:, :, 0].std(dim=1)` on the
    float32 tensor it feeds the model, and torch.std is UNBIASED by default
    (correction=1) while numpy's .std() is biased (ddof=0).  The two differ by
    sqrt(n/(n-1)) -- 0.78% at W=64 -- and that lands straight in the decode
    `y = z * wstd + wmean`.

    So this calls torch.std on a float32 copy rather than writing
    `np.std(..., ddof=1)`: those agree to 1.9e-9, but only the former is the
    SAME operator on the SAME dtype as the training path, which is the property
    we actually want.  Every evaluation path should call this instead of
    re-deriving the scale, which is how the conventions drifted apart.
    """
    return float(torch.as_tensor(np.asarray(x, dtype=np.float32)).std())


def decode_pred(pred, wmean, wstd, mode=None, ylast=None):
    """Invert the training target -- the one place the four decodes differ.

    z-score target:  tgt = (y - wmean) / wstd  ->  y = pred * wstd + wmean
    absolute target: tgt = y                   ->  y = pred
    anchor target:   tgt = y - wmean           ->  y = pred + wmean
    anchor_last:     tgt = y - ylast           ->  y = pred + ylast

    Getting this wrong is silent, not loud: applying the z-score rule to an
    absolute-target model multiplies a capacity-scale number (~0.85) by the
    window std (~0.018), so the model's own contribution is ~2% of the output
    and the decoded trajectory is essentially the true window mean.  Every
    seed then reports the same crossing, which is what exposed it -- see
    docs/ar_freeze_findings.md section 7.

    `mode=None` means "whatever this process is training", which is the right
    answer on the training path; callers evaluating a checkpoint from another
    mode must pass it explicitly (usually from the checkpoint's `tgt_mode`).
    """
    m = mode if mode is not None else TGT_MODE
    if m == "abs":
        return pred
    if m == "anchor":
        return pred + wmean
    if m == "anchor_last":
        assert ylast is not None, "anchor_last needs the window's last value"
        return pred + ylast
    return pred * wstd + wmean


def eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah, abs_target=None):
    model.eval()
    th = (eol_ah - lo) / (hi - lo + EPS)
    rows = []
    for tc in test_cells:
        seq = (caps[tc] - lo) / (hi - lo + EPS)
        seg_p = []
        with torch.no_grad():
            for i in range(sp, len(seq)):
                win = seq[i - W:i, None]
                cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
                wmean = float(win[:, 0].mean())
                # torch.std on the tensor the model is actually fed -- the same
                # operator and dtype the training target's scale came from.
                wstd = float(cin[:, :, 0].std(dim=1)) + EPS
                seg_p.append(decode_pred(float(model(cin).item()), wmean, wstd,
                                         abs_target, ylast=float(win[-1, 0])))
        seg_p = np.array(seg_p)
        tv = seq[sp:]
        n = min(len(tv), len(seg_p))
        tv, seg_p = tv[:n], seg_p[:n]
        trul = true_rul(tv, th)
        prul = true_rul(seg_p, th)
        rows.append({
            "TRUL": trul, "PRUL": prul, "AE": abs(trul - prul),
            "MAE": float(np.mean(np.abs(tv - seg_p))),
            "RMSE": float(np.sqrt(np.mean((tv - seg_p) ** 2))),
            "R2": float(1 - np.sum((tv - seg_p) ** 2) /
                        (np.sum((tv - tv.mean()) ** 2) + EPS)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None,
                    help="Override SP list; default uses load_series")
    ap.add_argument("--early-stop", action="store_true",
                    help="official-style 0.8/0.2 row-tail validation + "
                         "patience-10 best-checkpoint selection (ckpt "
                         "saved as SP{sp}_seed{n}_es.pt)")
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
    ds = args.dataset

    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    if args.sps:
        sps = args.sps
    caps = {c: caps[c].astype(np.float32) for c in caps}
    test_cells = [test_cell] if isinstance(test_cell, str) else list(test_cell)
    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = MIT_TEST_CELLS

    os.makedirs(f"../checkpoints/per_sp/{ds}", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    # The readout is part of the identity of a result file, not a detail of it:
    # a rate-head run with TGT_MODE=abs writes DIFFERENT numbers from a
    # free-head absolute run, so sharing per_sp_train_abs.json would overwrite
    # one experiment's only record with another's.  Keep them apart.
    _h = "" if READOUT == "last" else (
        "rate" if READOUT == "phys_ir" else READOUT)
    if BATCH != 64:
        _h = f"{_h}b{BATCH}" if _h else f"b{BATCH}"
    _head = (_h + "_") if _h else ""
    if TGT_MODE != "zscore":
        out_path = f"results/per_sp_train_{_head}{TGT_MODE}.json"
    elif AR_LOSS:
        out_path = f"results/per_sp_train_{_head}{AR_TAG}.json"
    elif SCHED_SAMPLE > 0:
        ss = "ss" if SCHED_K == 8 else f"ssk{SCHED_K}"
        out_path = f"results/per_sp_train_{_head}{ss}.json"
    else:
        out_path = f"results/per_sp_train{('_' + _head[:-1]) if _head else ''}.json"
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))

    # normalization lo/hi from the train split (non-test cells only)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    assert not (TGT_MODE != "zscore" and AR_LOSS), "AR loss is a z-score-mode option"

    for sp in sps:
        # train set: non-test cells full + test cell cycles < SP
        if AR_LOSS or SCHED_SAMPLE > 0:
            # XK/KM carry each window's next AR_KMAX true values.  The
            # continuation is capped at the same limit the windows obey, so
            # the test cell's windows never see a cycle at or past SP.
            Xtr, Ytr, XKtr, KMtr = build_windows(
                caps, train_cells, lo, hi, W, with_future=True, kmax=AR_KMAX)
            Xts, Yts, XKts, KMts = build_windows(
                caps, test_cells, lo, hi, W, max_cycle=sp, with_future=True,
                kmax=AR_KMAX)
            XK_all = np.vstack([XKtr, XKts])
            KM_all = np.vstack([KMtr, KMts])
        else:
            Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
            Xts, Yts = build_windows(caps, test_cells, lo, hi, W,
                                     max_cycle=sp)
            XK_all = KM_all = None
        X_all = np.vstack([Xtr, Xts])
        Y_all = np.concatenate([Ytr, Yts])
        n_tr = len(Xtr)

        for seed in range(args.start_seed, args.start_seed + args.seeds):
            skey = str(seed)
            # A non-zscore run is a SEPARATE model that must not collide with
            # the z-score checkpoints, so the resume-from-JSON dedup is off.
            if TGT_MODE == "zscore" and skey in out.setdefault(ds, {}).get(str(sp), {}):
                print(f"{ds} SP{sp} seed{seed}: SKIP (already in JSON)", flush=True)
                continue
            if args.early_stop:
                model, best_ep = train_one_earlystop(
                    seed, X_all, Y_all, W, patience=args.patience)
                suffix = _mode_suffix(True)
                tag = f"best_ep={best_ep}"
            else:
                # save_prefix turns on train_one's PROG_SAVE checkpoints
                # (SP{sp}_seed{n}{suffix}_ep{NNN}.pt).  Without it a 100-epoch
                # run yields nothing to evaluate until it is over -- which is
                # exactly the wait this experiment cannot afford, because the
                # question is whether the rollout degrades gracefully and that
                # can be answered from a part-trained model.
                ck_name = f"SP{sp}_seed{seed}{_mode_suffix(False)}"
                model = train_one(
                    seed, X_all, Y_all, W, XK_all, KM_all,
                    save_prefix=f"../checkpoints/per_sp/{ds}/{ck_name}",
                    save_meta=dict(lo=lo, hi=hi, sp=sp, eol_ah=eol_ah,
                                   test_cells=test_cells,
                                   train_cells=train_cells,
                                   abs_target=ABS_TARGET, tgt_mode=TGT_MODE,
                                   sched_sample=SCHED_SAMPLE))
                suffix = _mode_suffix(False)
                tag = f"epochs={EPOCHS}"
            torch.save(
                {"state_dict": model.state_dict(), "seed": seed,
                 "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": eol_ah,
                 "test_cells": test_cells, "train_cells": train_cells,
                 # records which decode this checkpoint needs; every _abs
                 # checkpoint written before 2026-09-15 lacks the key and must
                 # be told (eval_sp_batched(abs_target=True))
                 "abs_target": ABS_TARGET, "tgt_mode": TGT_MODE},
                f"../checkpoints/per_sp/{ds}/SP{sp}_seed{seed}{suffix}.pt")
            rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah)
            out.setdefault(ds, {}).setdefault(str(sp), {})[str(seed)] = rows
            print(f"{ds} SP{sp} seed{seed} ({tag}): "
                  f"AE={[r['AE'] for r in rows]} "
                  f"MAE={np.mean([r['MAE'] for r in rows]):.4f} "
                  f"R2={np.mean([r['R2'] for r in rows]):.4f} "
                  f"trainN={n_tr}+{len(Xts)}", flush=True)
            json.dump(out, open(out_path, "w"), indent=2)

    # 10-seed per-SP means
    print("=== per-SP means over seeds ===", flush=True)
    for sp in sps:
        vals = {m: [] for m in ["AE", "MAE", "RMSE", "R2", "TRUL", "PRUL"]}
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            for r in out[ds][str(sp)].get(str(seed), []):
                for m in vals:
                    vals[m].append(r[m])
        print(f"SP{sp}: AE={np.mean(vals['AE']):.2f} "
              f"MAE={np.mean(vals['MAE']):.4f} "
              f"RMSE={np.mean(vals['RMSE']):.4f} "
              f"R2={np.mean(vals['R2']):.4f}", flush=True)


if __name__ == "__main__":
    main()
