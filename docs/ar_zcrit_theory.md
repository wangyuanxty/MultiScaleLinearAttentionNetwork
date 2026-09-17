# Window-relative decoding: a self-consistency and stability theory

Why autoregressive rollout under the z-score decode freezes or runs away, why
the outcome differs between datasets, and when a longer window helps.  Every
claim is derived and then checked numerically, or measured directly.

---

## 0. The mechanism, without the algebra

Read this first; sections 1-9 are the same content with the derivation and the
verification tables.

### The decode

```
y_hat = z * sigma_window + mean_window
```

The model emits one scalar `z` per step.  The decode turns it into a capacity by
adding `z` times the window's standard deviation to the window's mean.  Two
structural consequences:

* the output magnitude is set by `sigma_window`, the spread of the `W` previous
  values -- which in rollout are the model's own previous outputs;
* `z` is expressed in units of that spread, so the same `z` means a different
  capacity step once the window changes.

The feedback is in the window: every output becomes an input to the next step.

### The fixed point

Take a window of `W` values declining by `rho` per step, most recent last.  Its
mean sits `rho*(W-1)/2` above the last value, and its standard deviation is

```
sigma_ramp = rho * A,        A = sqrt(W (W+1) / 12)
```

-- **the spread of a ramp is proportional to its slope.**  `A` is the *unbiased*
sd of `{0, ..., W-1}` because that is what the decode uses: `torch.std` defaults
to `correction=1`, the unbiased estimator.  The biased `sqrt((W^2-1)/12)`
differs by `sqrt(W/(W-1))` -- 0.79% at W=64 -- and pairing it with a
`torch.std` decode is the same mismatch as the evaluation bug recorded in
`eval-std-convention-bug`.  Everything below uses the unbiased one.

The next true value is `last - rho`, so a perfect model must emit

```
z_req = (y_next - mu)/sigma = -(W+1)/2 * (rho/sigma)
```

On a noise-free ramp the denominator is `sigma_ramp`, so `rho` appears in the
numerator and the denominator alike and divides out:

```
z_crit = -(W+1)/2 * (rho / (rho*A)) = -(W+1)/2/A = -sqrt( 3 (W+1) / W )
                                                          -1.7455 at W=64
```

**The step that is easy to skip: the cancellation is a substitution, not an
identity.**  On a real window `sigma = sqrt(sigma_ramp^2 + sigma_noise^2)`, the
denominator no longer carries `rho`, nothing divides out, and `r/sigma` becomes
a measurable property of the dataset (section 4).  The cancellation is what
makes `z_crit` dataset-independent -- and it holds only for `z_crit`, never for
`z_req`.

`z_crit` is the only constant `z` at which the recursion reproduces its own
window -- the trajectory keeps declining at whatever rate it already had.  It is
a repelling fixed point.  Emitting less than `|z_crit|` lands each step closer
to flat; the window flattens, `sigma` shrinks, and the same `z` now lands
flatter still, until the output is a constant.  Emitting more runs the same
argument in the other direction and the trajectory diverges.  Both outcomes
occur in the measured rollouts.

`z_crit` depends only on `W` and on the form of the decode -- not on the
battery, the chemistry, or the fade rate.

### Noise puts a perfect model on the collapsing side

A perfect one-step model, on a real window, emits

```
z_req = (true_next - mean) / sigma          -- which is the training label
```

For a falling window the numerator is about `-(W+1)/2` times the per-step fade.
The denominator is the **total** window spread, and the window is not a clean
line: it carries measurement noise.  Noise enters only here, and only upward,
so it makes `|z_req|` smaller and places `z_req` on the collapsing side of the
fixed point.

**A perfect model on any noisy signal collapses.**  The decode asks for a step
that is slightly too small, and the feedback amplifies "slightly".

### One number

```
delta = z_req - z_crit        distance from the fixed point at the launch window
```

The distance decays geometrically, so the rollout rate does too:

```
rate after n steps = rate * (1 - g*delta)^n        g = sqrt(3)/W
half-life          = 0.693 * W / (sqrt(3) * delta)
```

`delta` is the size of the error; the half-life is how many steps it takes to
matter.  Measured:

| dataset | W | delta | half-life | rollout |
|---|---|---|---|---|
| MIT EOL-50 | 64 | **0.0157** | 1630 steps | 19/20 cross, AE 1-8 |
| TJU EOL-50 | 64 | **0.0034** | 7640 | 8/10 cross, AE 24 |
| GOTION EOL-50 | 512 | **0.0272** | 7530 | **3/3 cross, AE 36-43** |
| CALCE EOL-50 | 64 | **0.2381** | 108 | 6/10 cross, AE 30-240 |

A half-life of thousands means the rollout outlives the horizon we ask about; a
half-life of ~100 means it dies inside it.  All four rows are at the same launch
point so they are comparable; section 4 gives both launch points per dataset.

### What moves delta: the window length, but only when the fade is straight

`sigma` grows with `W` -- a longer window spans more of the drop -- while the
noise does not, so a longer window moves `z_req` toward `z_crit`.
**Provided the fade is straight.**  If it is *convex* -- accelerating near end
of life, as CALCE's is -- a longer window reaches back into the flatter early
part and the fitted slope drops: CALCE's `r` falls 53% between W=64 and W=512,
which undoes the gain and then some.

```
straight fade (TJU, GOTION, PANASONIC)  ->  longer window closes delta  ->  helps
convex fade   (CALCE)                   ->  r collapses first          ->  stops helping
```

**GOTION is the clean demonstration**: at its natural `W=30` it is the worst
dataset in the study -- delta 1.577, half-life 8 steps, never crosses at any
launch point.  At `W=512` its delta drops to 0.027, inside the range where MIT
and TJU roll out, and it crosses at all three launch points with the error
**not growing with horizon** (AE 36/41/43 for launches 50/100/200 ahead).  Same
data, same decode, same training; only the window changed.

### Scope

* **It predicts** which (dataset, window, launch point) can roll out, from the
  series alone, before anything is trained -- and why the rest cannot.
* **It does not say** a bigger window rescues everything: the fade has to be
  straight enough for the mechanism to apply.
* **It is not a property of this model.**  `z_crit` follows from
  `y_hat = z*sigma + mean` and `W`; any model decoded that way has the same
  fixed point.

---

## 1. The decode under test

Training target and evaluation decode are one rule, inverted:

```
target:   z_t = (y_t - mu(w_t)) / sigma(w_t)
decode:   y_hat_t = z_t * sigma(w_t) + mu(w_t)
```

where `w_t = [y_hat_{t-W}, ..., y_hat_{t-1}]` in rollout — **the window is the
model's own output**, which is what makes this different from the teacher-forced
evaluation in Table A.

`mu` and `sigma` are the window mean and standard deviation.  In training
`sigma` is `torch.std` (unbiased); see `eval-std-convention-bug` for why the
evaluation side must use the same estimator.

Write `A = sqrt(W (W+1) / 12)`, the factor relating a linear ramp's step to its
standard deviation under the **unbiased** estimator -- the one the decode uses,
for the reason in section 0.

---

## 2. The self-consistency condition — where `z_crit` comes from

**Question.** For which constant `z` does the recursion reproduce its own
window, i.e. keep declining at a steady rate?

Let the trajectory decline linearly at rate `rho > 0` per step, `y_t = y_{t-1} - rho`.
Over a window of `W` values:

```
mu    = y_{t-1} + rho (W-1)/2          (older values are higher)
sigma = rho * A                         (unbiased sd of a pure ramp)
y_t   = y_{t-1} - rho                   A = sqrt(W (W+1) / 12)
```

Demanding `y_hat_t = y_t`:

```
z rho A + rho (W-1)/2 = -rho
z A = -(W+1)/2
```

```
        (W+1)        1                       sqrt(3 (W+1))
z_crit = - ----- ---------------   =  - sqrt( ------------- )
           2    sqrt(W(W+1)/12)                     W
```

Note **`rho` cancels**: the offset term, the scale term and the target step are
all proportional to `rho`.  So `z_crit` depends only on `W` and on the *form* of
the decode — not on the battery, the chemistry, or the fade rate.  W=64 gives
`z_crit = -1.745530`; W=30 gives `-1.760682`.

**Verified** (simulation: W=64, constant `z`, ramp input at rho = 6.7e-4, window
initialised as a ramp ending at 1.0, 3000 steps, unbiased sd throughout):

```
z = -1.79553   final y =  -636.80   steady drift = -1.52e+00   runaway
z = -1.76553   final y =   -13.25   steady drift = -1.50e-02   runaway (slow)
z = -1.74553   final y =    -1.01   steady drift = -6.700e-04  <- == input rho
z = -1.72553   final y =     0.40   steady drift = -2.93e-05   frozen
z = -1.44553   final y =     0.98   steady drift =  0.000e+00   frozen
```

At `z = z_crit` the steady drift is `-6.700e-04` against an input rate of
`6.7e-4` — the fixed point reproduces the input rate exactly.  Both sides are
repelling: less negative than `z_crit` converges to a constant window (sigma
collapses, the model's contribution vanishes); more negative accelerates
without bound.

---

## 3. The stability law — how fast a deviation grows

Let the window's rate be `rho_t` and let the model emit `z = z_crit + delta`.
The naive reading of section 2 is `rho_{t+1}/rho_t = 1 - A delta`, because a
pure ramp at the *new* rate would have that ratio.  **That is wrong**, and the
error is large: it treats the window as if all `W` values already carried the
new rate, when in fact one step replaces exactly one value and the remaining
`W-1` are unchanged.  The window's fitted slope is therefore an average, and the
per-step change is damped by that averaging.

Measuring the true ratio (constant `z`, ramp input, compare the window slope
after one appending step against the input slope) gives a coefficient of
`1/(2A)`, not `A`:

| W | z_crit | measured gain `g` | `1/(2A)` |
|---|---|---|---|
| 16 | -1.785357 | 0.105021 | 0.105021 |
| 30 | -1.760682 | 0.056796 | 0.056796 |
| 64 | -1.745530 | 0.026854 | 0.026854 |
| 128 | -1.738803 | 0.013479 | 0.013479 |
| 256 | -1.735430 | 0.006753 | 0.006753 |
| 512 | -1.733741 | 0.003380 | 0.003380 |
| 1024 | -1.732896 | 0.001691 | 0.001691 |

The two columns now agree exactly at every `W`.  The measurement is one
appending step at `z = z_crit + d`, and it is the same at `d = 0.02` and
`d = 0.05` -- the law is first-order in `d`.  (The earlier version of this table
paired a measurement taken with the unbiased sd against a prediction using the
biased one; that mismatch, not the law, is why its columns disagreed by up to 6%
at small `W`.)

```
rho_{t+1} / rho_t = 1 - (z - z_crit) / (2A)          (verified, W = 16..1024)
```

Since `A -> W/sqrt(12)` for large `W`:

```
g := 1/(2A) ~ sqrt(3) / W                             (gain falls as 1/W)
```

**This is the load-bearing step for everything practical.**  The drift is slow
(`g = 0.0269` at W=64, not `A = 18.6`), so a small deviation is harmless over a
short horizon; and because `g ~ 1/W`, lengthening the window damps the drift
directly and not only through the deviation below.

---

## 4. What the data contributes: `z_required`

The model does not emit `z_crit`; it emits what it was trained to emit, and for
a real (noisy) window that is

```
z_req = (y_next - mu) / sigma       -- the training target itself
```

which for a ramp is `-(W+1)/2 * (r / sigma)`, with `sigma` now the **total**
window spread, `sqrt(sigma_noise^2 + (r A)^2)`.

**Noise only inflates `sigma`, so `|z_req| <= |z_crit|` always**: a perfect
one-step model on any noisy signal lands on the *freezing* side of the fixed
point.  Define the deviation and substitute `z_crit = -(W+1)/2 / A`:

```
delta = z_req - z_crit
      = -(W+1)/2 * (r/sigma) + (W+1)/2 * (1/A)
      = (W+1)/2 * (1/A - r/sigma)                    (>= 0)
```

The signed form makes the noise argument immediate: `delta >= 0` **iff**
`r/sigma <= 1/A` **iff** `sigma >= r*A` -- i.e. iff the window carries any
variance beyond its own slope.  `r` cancels only in the noise-free limit.

**Read the small values with care.**  `delta` is a difference of two numbers
that, for a well-behaved cell, agree to three digits: at TJU EOL-50 (W=64) they
are `1/A = 0.053709` and `r/sigma = 0.053606`, 0.19% apart.  So **when `delta`
is small its magnitude is fragile** -- a 0.2% change in either term moves it by
100%.  What is robust is the *sign* (`delta > 0` always, by the argument above)
and the *ordering across datasets*, which spans three orders of magnitude (TJU
0.003, CALCE 0.24, GOTION 1.58).  Do not quote a small `delta` to its own
apparent precision.  Every value below is reproduced by
`src/diag_regen_vs_curvature.py`, which prints `r/sigma` to six digits for
exactly this reason.

and the rate decays geometrically:

```
rho_n = rho_0 (1 - g delta)^n        half-life   n_half ~ 0.693 W / (sqrt(3) delta)
```

Measured on four datasets (launch at the stated cycle, before any fitting):

| dataset / launch | W | r/sigma | z_req | z_crit | delta | observed rollout |
|---|---|---|---|---|---|---|
| TJU EOL-50 | 64 | 0.053606 | -1.7422 | -1.7455 | **+0.0034** | AE 24 |
| MIT EOL-100 | 64 | 0.053362 | -1.7343 | -1.7455 | +0.0113 | AE ~20 |
| MIT EOL-50 | 64 | 0.053225 | -1.7298 | -1.7455 | +0.0157 | **AE 1-8, 19/20 crossed** |
| TJU EOL-100 | 64 | 0.052929 | -1.7202 | -1.7455 | +0.0254 | AE 58.6 |
| CALCE EOL-100 | 64 | 0.048548 | -1.5778 | -1.7455 | +0.1677 | AE 88 |
| CALCE EOL-50 | 64 | 0.046383 | -1.5075 | -1.7455 | +0.2381 | AE 119 |
| GOTION EOL-100 | 30 | 0.047387 | -0.7345 | -1.7607 | +1.0262 | 0/10 crossed |
| GOTION EOL-50 | 30 | 0.011847 | -0.1836 | -1.7607 | +1.5771 | 0/10 crossed |

Half-lives from the law: MIT 1630 (EOL-50) and 2280 (EOL-100) steps, no effect
within a 100-step horizon; CALCE 108 and 153 steps, dominant within 50-100;
GOTION 8 and 12 steps, frozen on contact.
The ordering of the last column follows the ordering of `delta`.

**`r/sigma` is scale-invariant** — multiplying the series by any constant scales
`r` and `sigma` together — so no renormalisation of the input can move `delta`.
What moves it is `W`.

---

## 5. The window length, and when it helps

The derivation suggests lengthening `W` should work: `sigma_trend = r A ~
r W / sqrt(12)` grows with `W` while `sigma_noise` does not, so `z_req` should
move toward `z_crit`; and `g ~ 1/W` damps whatever drift remains.  As
`W -> inf`, both `z_req` and `z_crit` tend to `-sqrt(3)`, so `delta -> 0`.

**Whether that happens depends on one thing the derivation holds fixed: `r`.**
The formula treats `r` as the cell's fade rate, but what enters `z_req` is the
slope FITTED over the `W`-window -- and those coincide only if the fade is
straight across the window.

Measured on CALCE CS2_35, straight from the true series, no model involved
(`delta` is scale-invariant, so this needs no checkpoint):

| W | launch | r (per cycle) | sigma | z_req | z_crit | delta |
|---|---|---|---|---|---|---|
| 64 | EOL-50 | 5.912e-04 | 0.01275 | -1.5075 | -1.7455 | **+0.2381** |
| 128 | EOL-50 | 5.906e-04 | 0.02278 | -1.6725 | -1.7388 | **+0.0663** |
| 256 | EOL-50 | 4.758e-04 | 0.03771 | -1.6216 | -1.7354 | **+0.1139** |
| 512 | EOL-50 | 2.805e-04 | 0.04503 | -1.5979 | -1.7337 | **+0.1359** |

**`sigma` grows with `W` exactly as predicted; `r` falls by 53%.**  CALCE's fade
is convex -- accelerating toward end of life -- so a longer window reaches back
into the flatter early part.  The two effects compete and `r` wins, so `delta`
is **not monotone**: it drops from W=64 to W=128 and then climbs back.

A dataset whose fade is close to straight behaves the other way.  TJU, same
launch:

| W | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| r | 7.421e-04 | 7.316e-04 | 7.383e-04 | 7.055e-04 |
| delta (EOL-50) | +0.0108 | +0.0034 | +0.0062 | +0.0044 |
| half-life | 1190 | 7640 | 6200 | 11700 |

`r` moves 5% across a 4x window, so `sigma` does all the work.  Note that the
`delta` values no longer fall monotonically: they sit in a band of width
0.003-0.011, and the term differences driving them are 6.5e-4, 1.0e-4, 1.3e-4,
6.8e-5 -- at W=64 and W=96 the two terms agree to four digits, so their ordering
is not resolvable (section 4).  What survives is the claim that matters: **TJU is
one to two orders of magnitude closer to the fixed point than CALCE at every
`W`**, so it never needed a longer window in the first place.

GOTION is the dataset that does need one -- a 17x window:

| GOTION W | 30 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| delta (EOL-50) | +1.577 | +0.352 | +0.178 | +0.135 | **+0.027** |

and the rollout goes from **never crossing at any launch point** (W=30) to
**crossing at all three** with AE 36/41/43 that does not grow with horizon
(W=512).  Same data, same decode, same training schedule -- only the window.

So the effect is real, and whether it applies is itself a measurable property of
the data:

```
r_W / r_64 near 1   (straight fade)  ->  delta keeps falling  ->  helps
r_W / r_64 << 1     (convex fade)    ->  delta turns around   ->  stops helping
```

**CALCE is the convex case and the only dataset in this study that a longer
window does not rescue.**

---

## 6. What the half-life criterion is for

`n_half = 0.693 W / (sqrt(3) delta)` answers one question: **given this
deviation, how many steps before the trajectory has drifted off its launch
behaviour?**  It is a diagnostic, and it is useful in that role -- it is what
separates "this rollout outlives the horizon" (MIT: 1630 steps) from "this one
dies inside it" (CALCE: 108 steps, against a horizon of 50-200).

As a *prescription* it only works when `delta` itself shrinks with `W`, which
section 5 shows is a property of the fade shape rather than a given:

* straight fade (GOTION, PANASONIC): `delta` falls, so the criterion predicts
  the improvement that is then measured.  (TJU is straight too, but it is
  already one to two orders of magnitude below CALCE at every `W`, so the
  criterion says the window was never its constraint.)
* convex fade (CALCE): `r` collapses before `sigma` can pay for it, `delta`
  turns around, and the criterion correctly predicts that lengthening the window
  stops helping.

The criterion is therefore both diagnostic and prescriptive, with the second
role conditional on the fade shape -- and the condition is itself measurable
from the data before any training (`r_W / r_64`).

---

## 7. The model does forecast — controls

A criterion is only meaningful if the failure it predicts is a failure.  Two
controls on MIT (dose checkpoints, launch EOL-50 / EOL-100, 1000 steps):

| launch | model | linear extrapolation | constant (window-last) |
|---|---|---|---|
| EOL-50, cell5 | **490** (true 489) | 534 | never crosses |
| EOL-50, cell47 | **718** (true 726) | 766 | never crosses |
| EOL-100, cell5 | **523** (true 489) | 687 | never crosses |
| EOL-100, cell47 | **728** (true 726) | 853 | never crosses |

The model beats linear extrapolation by 5-10x at both horizons, and constant
extrapolation never crosses.  So the crossings are forecasts, not a trend
continuation and not an artefact any method would produce.

Training-set size does not enter: `N=1` and `N=4` give the same AE (4.5), and
`N=20` is no better inside the usable horizon.  The dose experiment's training
sets are nested prefixes, so this is a within-dataset comparison.

---

## 8. Scope and limitations

* **The analysis assumes a constant `z`.**  The real model emits a function of
  the window, so sections 2-3 describe the dynamics the model's output *would*
  produce if held fixed; they are the right frame for "is this decode
  self-consistent", not a closed-form prediction of a particular checkpoint.
* **`delta` is computed on the true launch window**, i.e. as an initial
  condition.  Once the rollout starts, the window is self-generated and `r/sigma`
  drifts; the half-life is therefore an estimate of the time to lose the launch
  behaviour, not a bound on the rollout.
* **`g = 1/(2A)` is verified for constant `z` and a noiseless ramp.**  With noise
  the effective `d sigma / d rho` is smaller and the gain is smaller still; the
  measured four-dataset ordering uses the noiseless formula for `delta` and is
  therefore conservative.
* **Curvature is ignored.**  Real fade is convex near end of life, which is
  precisely the part the model has to get right to beat linear extrapolation.
  The controls in section 7 show it does, but nothing here models it.
* **The estimator convention is load-bearing.**  `A` and the measured `sigma`
  must BOTH be the unbiased one, matching the decode's `torch.std`.  Mixing them
  shifts `delta` by `(W+1)/2 * (1/A_biased - 1/A_unbiased)` -- 0.0138 at W=64,
  which is four times the whole of TJU's `delta` there.  An earlier version of
  this document did exactly that; the tables above are consistent.  Section 4
  covers the fragility this implies when `delta` is small.

---

## 9. What this implies for the paper

1. **The decode, not the model, sets whether rollout is possible.**  `z_crit`
   is a property of `y_hat = z sigma + mu` and `W` alone.
2. **A rollout section can state a boundary rather than an apology**: report the
   deviation criterion, the datasets it excludes, and the controls that show the
   forecasts it does allow are real.
3. **A longer window is the fix where the fade is straight, and the data says
   which case you are in.**  GOTION at W=512 is the demonstration: delta
   1.577 -> 0.027, never-crossing -> 3/3 crossing.  CALCE is the counter-case,
   and `r_W / r_64` (5% for TJU, 53% for CALCE) predicts which before anything
   is trained.  Do not state it unconditionally in either direction.
4. **The evaluation-protocol fix (unbiased `sigma`)** changes no reported number
   beyond the quoted uncertainties; it is recorded in
   `eval-std-convention-bug` and needs no place in the text.  The same convention
   applies to `A` here -- see section 0.
