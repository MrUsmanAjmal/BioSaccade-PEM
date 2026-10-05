"""Dataset loading and checks."""
from pathlib import Path
from dataclasses import asdict, dataclass
import hashlib, json, os, re, time, urllib.request, urllib.error, zipfile
import cv2
import numpy as np

DATASETS = {
    'DAVIS': {'url': 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip',
              'sha256': 'e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492',
              'source': 'https://davischallenge.org/davis2017/code.html'},
    'REDS': {'url': 'https://huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip',
             'source': 'https://seungjunnah.github.io/Datasets/reds.html',
             'split': 'val_sharp, 30 clips x 100 frames'},
    'REDS_train': {'url': None,
                   'source': 'https://seungjunnah.github.io/Datasets/reds.html',
                   'split': 'train_sharp, 240 clips x 100 frames',
                   'note': 'Supplied as a local copy. Replaces Vimeo-90K (official download unavailable). '
                           'No model in this study is trained on REDS, so it is unseen external test data.'},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(2 ** 20), b''):
            h.update(part)
    return h.hexdigest()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
    os.replace(tmp, p)


def immutable_json(path, obj):
    p = Path(path)
    if p.exists():
        if json.loads(p.read_text()) != json.loads(json.dumps(obj, default=str)):
            raise RuntimeError(f'Locked inputs changed: {p}. Choose a new experiment directory.')
    else:
        atomic_json(p, obj)
    return obj


def download(url, path, expected=None):
    """Download with resume support."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    record = path.with_name(path.name + '.download.json')
    if path.exists():
        observed = sha256(path)
        old = json.loads(record.read_text()) if record.exists() else {}
        check = expected or old.get('sha256')
        if check and observed != check:
            raise ValueError(f'Content changed or corrupt: {path}')
        if not record.exists():
            atomic_json(record, {'url': url, 'sha256': observed, 'bytes': path.stat().st_size,
                                 'verification': 'upstream-pinned' if expected else 'first-observed'})
        return path
    partial = path.with_name(path.name + '.partial')
    for attempt in range(3):
        try:
            size = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
                       'Accept': '*/*', 'Referer': 'https://toflow.csail.mit.edu/'}
            if size:
                headers['Range'] = f'bytes={size}-'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                resumed = size > 0 and r.status == 206
                if resumed and not r.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
                    raise RuntimeError('Invalid server resume offset')
                total = size if resumed else 0; announced = total
                with partial.open('ab' if resumed else 'wb') as f:
                    while True:
                        part = r.read(2 ** 20)
                        if not part:
                            break
                        f.write(part); total += len(part)
                        if total - announced >= 128 * 2 ** 20:
                            print(path.name, round(total / 2 ** 30, 2), 'GiB downloaded', flush=True)
                            announced = total
            h = sha256(partial)
            if expected and h != expected:
                partial.unlink(missing_ok=True)
                raise ValueError(f'Checksum mismatch: {path.name}')
            os.replace(partial, path)
            atomic_json(record, {'url': url, 'sha256': h, 'bytes': path.stat().st_size,
                                 'verification': 'upstream-pinned' if expected else 'first-observed'})
            return path
        except Exception:
            if attempt == 2:
                raise


def extract_dataset(archive, destination, dataset):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    stamp = destination / 'extraction.json'
    fingerprint = {'archive_sha256': sha256(archive), 'dataset': dataset}
    if stamp.exists() and json.loads(stamp.read_text()) == fingerprint:
        return destination
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            rel = Path(entry.filename)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive member')
            if entry.is_dir():
                continue
            name = entry.filename
            if dataset == 'DAVIS':
                keep = '/JPEGImages/480p/' in name or '/ImageSets/2017/' in name
            else:
                keep = name.lower().endswith(('.png', '.jpg'))
            if keep:
                p = destination / rel; p.parent.mkdir(parents=True, exist_ok=True)
                # Opening and reading a member verifies its CRC, including after resume.
                with z.open(entry) as src, p.open('wb') as dst:
                    while True:
                        b = src.read(2 ** 20)
                        if not b: break
                        dst.write(b)
    atomic_json(stamp, fingerprint)
    return destination


@dataclass(frozen=True)
class Clip:
    dataset: str
    clip_id: str
    source_group: str
    role: str
    paths: tuple
    content_sha256: str


def _clip(dataset, name, group, role, files):
    paths = tuple(str(p.resolve()) for p in files)
    h = hashlib.sha256()
    for p in files:
        h.update(sha256(p).encode())
    return Clip(dataset, name, f'{dataset}:{group}', role, paths, h.hexdigest())


def discover_davis(root, split, frames):
    roots = list(Path(root).rglob('JPEGImages'))
    if len(roots) != 1:
        raise ValueError(f'Expected one DAVIS JPEGImages directory in {root}')
    p = roots[0] / '480p'; clips = []
    role_map = {'fit': 'fit', 'development': 'development', 'reserved': 'legacy_test', 'test': 'legacy_test'}
    for key, names in split.items():
        for name in names:
            files = sorted((p / name).glob('*.jpg'))[:frames]
            if len(files) < 7:
                raise ValueError(f'Missing DAVIS clip: {name}')
            clips.append(_clip('DAVIS', name, name, role_map[key], files))
    return clips


def discover_reds(root, frames=100, limit=None, dataset='REDS', expected=None):
    """One REDS split folder (val_sharp or train_sharp) -> one clip per subfolder.

    Train and val clip folders both start at 000, so each split is registered as
    its own dataset ('REDS' for val_sharp, 'REDS_train' for train_sharp); clip IDs
    and source groups therefore never collide. `expected` is the number of clips
    the split must contain; `limit` keeps the first clips by name (pilot runs).
    """
    parents = sorted({p.parent for p in Path(root).rglob('*.png')}, key=lambda p: p.name)
    if not parents:
        raise ValueError(f'{dataset}: no PNG frames found in {root}')
    if len({p.name for p in parents}) != len(parents):
        raise ValueError(f'Ambiguous {dataset} directories in {root}; point to one split folder (val_sharp or train_sharp) only')
    if expected is not None and len(parents) != expected:
        raise ValueError(f'Incomplete {dataset} split: found {len(parents)} / {expected} clips in {root}')
    clips = []
    for p in (parents if limit is None else parents[:limit]):
        files = sorted(p.glob('*.png'))[:frames]
        if len(files) < 7:
            raise ValueError(f'{dataset} clip {p.name} has fewer than 7 frames')
        clips.append(_clip(dataset, p.name, p.name, 'external_test', files))
    return clips


def validate_manifest(clips):
    if not clips:
        raise ValueError('Empty dataset manifest')
    keys = [(c.dataset, c.clip_id) for c in clips]
    assert len(set(keys)) == len(keys), 'Duplicate clip identity'
    roles, hashes = {}, {}
    for c in clips:
        roles.setdefault(c.source_group, set()).add(c.role)
        hashes.setdefault(c.content_sha256, []).append(c)
    for name, role in roles.items():
        assert len(role) == 1, f'Source crosses partitions: {name}'
    for vals in hashes.values():
        assert len({v.role for v in vals}) == 1, 'Identical content crosses partitions'
    duplicates = [[v.clip_id for v in vals] for vals in hashes.values() if len(vals) > 1]
    # Exact duplicates within evaluation get one canonical source group so
    # inference never counts their duplicated clips as independent sources.
    from dataclasses import replace
    parent={c.source_group:c.source_group for c in clips}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    for vals in hashes.values():
        roots=sorted({find(c.source_group) for c in vals})
        for g in roots:parent[g]=roots[0]
    clips = [replace(c, source_group=find(c.source_group)) for c in clips]
    return clips, {'exact_duplicate_sets': duplicates, 'source_separation': True,
                   'reds_grouping': 'each REDS clip folder is one source cluster; REDS publishes no raw-video identity mapping, so near-duplicate scenes across clips cannot be excluded'}


def load_clip(clip, cfg):
    frames = []
    for path in clip.paths[:cfg.frames]:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f'Cannot decode image: {path}')
        im = im[..., ::-1]; h, w = im.shape[:2]
        if h < cfg.height or w < cfg.width:
            raise ValueError(f'{clip.clip_id}: source {h}x{w} smaller than requested crop {cfg.height}x{cfg.width}')
        y, x = (h - cfg.height) // 2, (w - cfg.width) // 2
        frames.append(im[y:y+cfg.height, x:x+cfg.width].astype(np.float32) / 255.)
    return np.stack(frames)


def synthetic_frames(cfg, seed=1, cut=False):
    """A software test fixture, never a natural-video benchmark substitute."""
    rng = np.random.default_rng(seed)
    texture = rng.uniform(0, 1, (cfg.height, cfg.width, 3)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), .5)
    out = []
    for t in range(cfg.frames):
        im = np.roll(texture, (t, 2*t), axis=(0, 1)).copy()
        if cut and t >= cfg.frames // 2:
            im = 1 - im
        out.append(im)
    return np.stack(out)
