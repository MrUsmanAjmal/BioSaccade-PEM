"""Recent SR baselines."""
from pathlib import Path
import importlib.util
import hashlib
import time
import numpy as np
from r9_data import download, atomic_json, sha256

COMMIT = 'b69fb668c0362deb696eecbfdebfaa5c2fcdfcb4'
BASE = f'https://raw.githubusercontent.com/Amazingren/NTIRE2025_ESR/{COMMIT}/'
SPECS = {
 'NanoSR2025': dict(code='models/team07_NanoSR.py', weights='model_zoo/team07_NanoSR.pth',
                   code_sha='a001a75884786651849f1561de6ba0767ecf00cce43c20bc621de9b428a5d87e',
                   weights_sha='7c584c8ad7c79ee2b20e923ed610f0b40992ac66e7e8edafd19b45c9a0ba3e49',
                   cls='NanoSR_inference', args=[3, 3], kwargs={}, state=None),
 'SPANF2025': dict(code='models/team24_SPANF.py', weights='model_zoo/team24_spanf.pth',
                  code_sha='f4ae70c75cc51cdba090c594dbd52525b61870d3ede36099e43d4a38bbe1fbf1',
                  weights_sha='1508c566008654646bea1edff537c83dd7a6fe8f6ffda3f25f9eb7ba09cbbcd3',
                  cls='SPANF', args=[3, 3], kwargs={'upscale': 4, 'feature_channels': 32}, state=None),
 'SCMSR2025': dict(code='models/team16_SCMSR.py', weights='model_zoo/team16_SCMSR.pth',
                  code_sha='cd78b85b1f83aafd41085a2a5c3f8dc2e25ecf24f1f3b95cce921ebc5b746764',
                  weights_sha='68a8f2c24743050bc510c813f065d3379671cb2bdd6ee83c0bcb15f14f817deb',
                  cls='SCMSR', args=[], kwargs={}, state='params_ema'),
}


class OfficialSR:
    def __init__(self, name, root, device='auto'):
        import torch
        self.torch = torch; self.name = name
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        spec = SPECS[name]; root = Path(root) / name; root.mkdir(parents=True, exist_ok=True)
        download(BASE+'LICENSE',root/'LICENSE','81032fbef39a967bdc9dc7f5aaaa3b65fe589c931c0cd38c473ecdf493b0e17a')
        source = download(BASE + spec['code'], root / Path(spec['code']).name, spec['code_sha'])
        weights = download(BASE + spec['weights'], root / Path(spec['weights']).name, spec['weights_sha'])
        code = source.read_text(encoding='utf-8'); patch = 'none'
        if name == 'SPANF2025':
            line = '        self.cuda()(torch.randn(1, 3, 256, 256).cuda())'
            if code.count(line) != 1:
                raise RuntimeError('Official SPAN-F warm-up changed; review the pinned source')
            code = code.replace(line, '        # Constructor-only CUDA warm-up omitted for CPU portability.')
            patch = 'remove_constructor_cuda_warmup_only'
        portable = root / 'portable_model.py'; portable.write_text(code, encoding='utf-8')
        module_spec = importlib.util.spec_from_file_location('official_' + name, portable)
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        model = getattr(module, spec['cls'])(*spec['args'], **spec['kwargs'])
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if spec['state'] is not None:
            state = state[spec['state']]
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.parameters = int(sum(p.numel() for p in model.parameters()))
        self.provenance = dict(name=name, year=2025, repository='https://github.com/Amazingren/NTIRE2025_ESR',
            commit=COMMIT, source_sha256=sha256(source), portable_source_sha256=sha256(portable),
            checkpoint_sha256=sha256(weights), patch=patch, parameter_count=self.parameters,
            device=self.device, scale=4, precision='float32', input='RGB [0,1]',
            adaptation='current LR preview; optional identical causal residual memory and paid patch',
            pretrained_training_data='external author checkpoint; see NTIRE 2025 report; cross-dataset overlap not fully auditable')
        atomic_json(root / 'model_provenance.json', self.provenance)
        self.warmup()

    causal = True
    temporal = False

    def predict_sequence(self, previews):
        """Frame-by-frame single-image SR; trivially causal."""
        preds, times = [], []
        for p in previews:
            y, ms = self.predict(p)
            preds.append(y); times.append(ms)
        return preds, times

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def warmup(self):
        for _ in range(2):
            self.predict(np.full((32, 32, 3), .5, np.float32))

    def predict(self, preview):
        torch = self.torch
        self.sync(); wall_start = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(np.clip(preview, 0, 1).transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        h, w = x.shape[-2:]
        # Preserve each author's native padding and input normalization.
        # SCMSR performs its own reflection padding inside forward_origin.
        self.sync(); start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(x)
        self.sync(); ms = 1000 * (time.perf_counter() - start)
        if isinstance(out, (tuple, list)):
            raise TypeError('Unexpected official model output')
        y = out[0, :, :h*4, :w*4].detach().float().cpu().numpy().transpose(1, 2, 0)
        if y.shape != (h*4, w*4, 3) or not np.isfinite(y).all():
            raise ValueError('Invalid pretrained model prediction')
        # Charge conversion and device transfers as well as model execution.
        y = np.clip(y, 0, 1).astype(np.float32)
        return y, 1000*(time.perf_counter()-wall_start)


# ---------------------------------------------------------------------------
# Video super-resolution baseline: BasicVSR++ (CVPR 2022), official OpenMMLab
# Vimeo-90K BI checkpoint. It is trained on the Vimeo-90K *training* split only,
# so REDS (train_sharp and val_sharp) and DAVIS are unseen.
# (The REDS4 checkpoint is deliberately NOT used: it is trained on REDS train+val.)
# ---------------------------------------------------------------------------
VSR_SPEC = dict(
    name='BasicVSRpp2022',
    urls=['https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/'
          'basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth',
          'https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/'
          'basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi_20210305-4ef437e2.pth'],
    sha256_prefix='4ef437e2',
    paper='K. C. K. Chan et al., BasicVSR++, CVPR 2022',
    training_data='Vimeo-90K septuplet training split, bicubic x4 degradation')


def _load_vsr_state(path):
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    gen = {k[len('generator.'):]: v for k, v in state.items() if k.startswith('generator.')}
    if not gen:
        gen = {k: v for k, v in state.items() if not k.startswith('step_counter')}
    return gen


class TemporalWindowSR:
    """Shared-weight video SR wrapper.

    mode='causal': the prediction for frame t uses only previews max(0,t-W+1)..t,
    the same information available to every other online method.
    mode='offline': one bidirectional pass over the whole clip (reads FUTURE
    previews); reported only as a non-causal reference.
    """
    temporal = True

    def __init__(self, net, provenance, device, mode='causal', window=7):
        assert mode in ['causal', 'offline'] and window >= 2
        import torch
        self.torch = torch; self.net = net; self.device = device
        self.mode = mode; self.window = window; self.causal = mode == 'causal'
        self.name = provenance['name'] + ('_causal' if self.causal else '_offline')
        self.parameters = provenance['parameter_count']
        self.provenance = dict(provenance, name=self.name, mode=mode,
                               window=(window if self.causal else 'full clip'),
                               causal=self.causal)

    def sync(self):
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize()

    def _run(self, stack):
        """stack: (t,h,w,3) float32 in [0,1] -> (t,4h,4w,3)."""
        torch = self.torch
        t, h, w, _ = stack.shape
        ph, pw = max(64, -(-h // 4) * 4), max(64, -(-w // 4) * 4)
        x = torch.from_numpy(np.ascontiguousarray(np.clip(stack, 0, 1).transpose(0, 3, 1, 2))).float()
        x = x.unsqueeze(0).to(self.device)
        if (ph, pw) != (h, w):
            x = torch.nn.functional.pad(x.view(t, 3, h, w), (0, pw - w, 0, ph - h), mode='replicate').view(1, t, 3, ph, pw)
        if t == 1:
            x = torch.cat([x, x], dim=1)
        with torch.inference_mode():
            y = self.net(x)[0, :t, :, :h * 4, :w * 4]
        y = y.float().cpu().numpy().transpose(0, 2, 3, 1)
        if not np.isfinite(y).all():
            raise ValueError('Invalid video SR prediction')
        return np.clip(y, 0, 1).astype(np.float32)

    def predict_sequence(self, previews):
        stack = np.stack([np.asarray(p, np.float32) for p in previews])
        if not self.causal:
            self.sync(); start = time.perf_counter()
            out = self._run(stack); self.sync()
            ms = 1000 * (time.perf_counter() - start) / len(stack)
            return list(out), [ms] * len(stack)
        preds, times = [], []
        for t in range(len(stack)):
            self.sync(); start = time.perf_counter()
            window = stack[max(0, t - self.window + 1):t + 1]
            preds.append(self._run(window)[-1]); self.sync()
            times.append(1000 * (time.perf_counter() - start))
        return preds, times


def load_video_sr(root, device='auto', window=7):
    """Download (or reuse) the official checkpoint and build causal + offline wrappers."""
    import torch
    from r9_vsr_arch import BasicVSRPlusPlusNet
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    folder = Path(root) / VSR_SPEC['name']; folder.mkdir(parents=True, exist_ok=True)
    path = None; errors = []
    local = sorted(folder.glob('*.pth'))
    for p in local:
        if sha256(p).startswith(VSR_SPEC['sha256_prefix']):
            path = p; break
    if path is None:
        for url in VSR_SPEC['urls']:
            try:
                candidate = download(url, folder / url.rsplit('/', 1)[1])
            except Exception as error:
                errors.append(f'{url}: {type(error).__name__}: {error}'); continue
            if sha256(candidate).startswith(VSR_SPEC['sha256_prefix']):
                path = candidate; break
            errors.append(f'{url}: SHA-256 prefix mismatch'); candidate.unlink()
    if path is None:
        raise RuntimeError('BasicVSR++ Vimeo-90K BI checkpoint unavailable. Download it from the mmagic '
                           'BasicVSR++ model zoo and place the .pth file in ' + str(folder) + '. ' + ' | '.join(errors))
    net = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7)
    state = _load_vsr_state(path)
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in ('spynet.mean', 'spynet.std')]
    if missing or unexpected:
        raise RuntimeError(f'BasicVSR++ checkpoint/architecture mismatch. Missing={missing[:8]} Unexpected={unexpected[:8]}')
    net = net.to(device).eval()
    provenance = dict(name=VSR_SPEC['name'], year=2022, paper=VSR_SPEC['paper'],
                      repository='https://github.com/open-mmlab/mmagic', checkpoint_file=path.name,
                      checkpoint_sha256=sha256(path), parameter_count=int(sum(p.numel() for p in net.parameters())),
                      training_data=VSR_SPEC['training_data'], device=device, scale=4, precision='float32',
                      implementation='mmagic-compatible PyTorch port with torchvision deform_conv2d; strict key check; behavioural sanity gate',
                      adaptation='current (and, for causal mode, past) LR previews; optional identical causal residual memory and paid patch')
    atomic_json(folder / 'model_provenance.json', provenance)
    causal = TemporalWindowSR(net, provenance, device, 'causal', window)
    offline = TemporalWindowSR(net, provenance, device, 'offline', window)
    causal.predict_sequence([np.full((64, 64, 3), .5, np.float32)] * 3)
    return {causal.name: causal, offline.name: offline}


def load_baselines(root, device='auto', include_video=True):
    models = {name: OfficialSR(name, root, device) for name in SPECS}
    if include_video:
        models.update(load_video_sr(root, device))
    return models


def baseline_sanity(models, clips_frames, factor=4):
    """Behavioural gate: each pretrained SR model must beat bicubic upsampling
    on clean, MATLAB-bicubic-downsampled held-out frames. A mis-ported or
    mis-loaded network fails loudly instead of producing a weak baseline."""
    import cv2
    from video_core import matlab_bicubic_down
    report = {}
    for name, model in models.items():
        gains = []
        for k, frames in enumerate(clips_frames):
            # Reproducible stochastic inference (SCMSR samples gumbel_softmax in eval mode).
            import torch
            from video_core import seed
            torch.manual_seed(seed('sanity', name, k))
            previews = [matlab_bicubic_down(f, factor) for f in frames]
            preds, _ = model.predict_sequence(previews)
            for f, p, y in zip(frames, previews, preds):
                up = np.clip(cv2.resize(p, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
                b = 8
                mse_sr = float(np.mean((y[b:-b, b:-b] - f[b:-b, b:-b]) ** 2))
                mse_bi = float(np.mean((up[b:-b, b:-b] - f[b:-b, b:-b]) ** 2))
                gains.append(10 * np.log10(mse_bi / max(mse_sr, 1e-12)))
        report[name] = dict(mean_psnr_gain_over_bicubic_db=float(np.mean(gains)), frames=len(gains),
                            passed=bool(np.mean(gains) > 0.0))
        if not report[name]['passed']:
            raise RuntimeError(f'Baseline sanity gate failed for {name}: {report[name]}')
    return report
