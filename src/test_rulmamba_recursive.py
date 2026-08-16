"""RUL-Mamba TRUE autoregressive K=1 RUL prediction.
Bypasses pytorch_forecasting — uses raw RULMamba encoder-decoder.
Protocol: pure capacity, K=1 single-step AR, feed pred back."""
import sys, torch, numpy as np, os, glob

torch.serialization.add_safe_globals([
    'pytorch_forecasting.data.encoders.EncoderNormalizer',
    'pytorch_forecasting.data.encoders.NaNLabelEncoder',
    'pytorch_forecasting.data.encoders.MultiNormalizer',
    'pytorch_forecasting.data.encoders.TorchNormalizer',
    'pytorch_lightning.callbacks.early_stopping.EarlyStopping',
    'numpy.core.multiarray.scalar',
])
_torch_load = torch.load
def _patched_load(*a, **kw):
    kw['weights_only'] = False
    return _torch_load(*a, **kw)
torch.load = _patched_load

sys.path.insert(0, r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_rul_mamba')
from Models.RULMamba import RULMambaNetModel


def run(dataset):
    """Load RUL-Mamba checkpoint, run K=1 autoregressive RUL."""
    if dataset == 'nasa':
        data = torch.load(os.path.join(sys.path[0], 'Data', 'NASA', 'Real_Data.pth'))
        test_caps = data['B0005']['capacity'].numpy().astype(np.float32)
        train_caps = np.concatenate([data[b]['capacity'].numpy() for b in ['B0006','B0007','B0018']]).astype(np.float32)
        W, EOL_AH, SPs = 30, 1.40, [50, 70, 90]
        ckpt_dir = os.path.join(sys.path[0], 'results_NASA_RUL_prediction_sl_30', 'B0005', 'RULMamba')
    else:
        print(f"Unknown dataset: {dataset}")
        return

    x_min, x_max = train_caps.min(), train_caps.max()
    test_norm = (test_caps - x_min) / (x_max - x_min)
    eol_norm = (EOL_AH - x_min) / (x_max - x_min)
    true_eol = int(np.argmax(test_caps < EOL_AH))
    print(f"RUL-Mamba {dataset}: {len(test_caps)} cycles, EOL@{EOL_AH}Ah={true_eol}")

    ckpts = glob.glob(os.path.join(ckpt_dir, '**', '*.ckpt'), recursive=True)
    if not ckpts:
        print("No checkpoint — need to train RUL-Mamba first")
        return
    ckpt = ckpts[0]
    print(f"Using: {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")

    model = RULMambaNetModel.load_from_checkpoint(ckpt).cuda().eval()
    core = model.network.cuda()  # RULMamba core

    for SP in SPs:
        window = test_norm[SP-W:SP]
        enc_in = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).cuda()

        # Encoder: context = last hidden state
        with torch.no_grad():
            enc_out = core.encoder(enc_in)
            context = enc_out[:, -1:, :]

        # Autoregressive: K=1, feed pred back
        dec_x = torch.tensor([[[window[-1]]]], dtype=torch.float32).cuda()
        preds = []
        for step in range(800):
            with torch.no_grad():
                v = core.var_enc(dec_x)
                for dec_block in core.dec:
                    v = dec_block(v, context)
                p = core.projection(v.squeeze(1)).item()
            preds.append(p)
            dec_x = torch.tensor([[[p]]], dtype=torch.float32).cuda()
            if p < eol_norm:
                break

        n = len(preds)
        if n < 800 and n >= 2:
            j = n - 2
            frac = (eol_norm - preds[j]) / (preds[j+1] - preds[j] + 1e-8) if (preds[j] >= eol_norm > preds[j+1]) else 0
            pe = SP + j + frac
        elif n < 800:
            pe = SP + n
        else:
            pe = -1
        ae = round(abs(true_eol - pe), 1) if pe >= 0 else -1
        print(f'SP={SP}: AE={ae} cycles, pred_EOL={pe:.1f}, steps={n}')

run('nasa')
