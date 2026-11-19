# %% [markdown]
# # PAPER023041 — дополнительные эксперименты v4
#
# **Как запустить:** Среда выполнения → Сменить среду выполнения → **T4 GPU** → Сохранить, затем **Среда выполнения → Выполнить все**. Разрешите доступ к Google Диску — к тому аккаунту, где лежит папка **PAPER023041_results** с результатами версии 3 (папка checkpoints с тремя сетями, raw_results.csv, sensitivity_v3.csv).
# Время: около **2–2.5 часов**. Если Colab отключится — снова «Выполнить все», работа продолжится с места остановки.
# Результаты версии 3 **не изменяются**; новое пишется в файлы *_v4. В конце пришлите **RESULTS_PAPER023041_v4.xlsx** из папки PAPER023041_results.
#
# Что считается:
# 1. Обучение сети с весом стиля 1 и выбор (вес стиля, α) на валидационных текстурах для метода **без B-сплайна** (новый предлагаемый вариант).
# 2. Абляция: та же сеть, обученная на **белом шуме** вместо процедурной основы (нужна ли процедурная часть).
# 3. Честный генеративный бейзлайн, который видит образец: **Stable Diffusion v1.5 + IP-Adapter**.
# 4. Предлагаемый метод на 512×512.
# 5. Разнообразие результатов разных сидов через **LPIPS** (со сплайном и без).

# %%
# ===== 1. НАСТРОЙКИ (менять не нужно) =====
TEST_MODE = False
CATEGORIES = ["woven", "braided", "stratified", "knitted", "marbled"]
N_TEST_PER_CAT, N_512_PER_CAT, N_VAL_PER_CAT = 15, 6, 3
SEEDS = (0, 1, 2)
RES_MAIN, RES_HIGH = 256, 512
ALPHA = 0.5
OPT_STEPS, LAMBDA_C, LAMBDA_TV = 150, 0.5, 1e-4
OPT_LAMBDA_S_GRID = [1e3, 1e4, 1e5, 1e6]
VGG_GRAM_STEPS = 250
FF_ITERS, FF_BATCH, FF_CROP, FF_LR = 4000, 8, 256, 1e-4
FF_STYLE_W_GRID = [3.0, 10.0, 30.0]
FF_W_NEW = 1.0
IP_SCALES = [0.4, 0.7, 1.0]
SD_STEPS = 25
FF_ALPHA_GRID = [0.25, 0.5, 0.75, 1.0]
N_STYLE_IMAGES = 2500
QUILT_PATCH, QUILT_OVERLAP = 32, 8
OUT_DIR = "/content/drive/MyDrive/PAPER023041_results"
if TEST_MODE:
    N_TEST_PER_CAT, N_512_PER_CAT, N_VAL_PER_CAT = 2, 1, 1
    SEEDS = (0, 1); RES_MAIN, RES_HIGH = 64, 96
    OPT_STEPS, VGG_GRAM_STEPS = 3, 3; OPT_LAMBDA_S_GRID = [1e3, 1e5]
    FF_ITERS, FF_BATCH, FF_CROP = 3, 2, 64; FF_STYLE_W_GRID = [3.0, 10.0]; FF_ALPHA_GRID = [0.5, 1.0]; IP_SCALES = [0.4, 1.0]; SD_STEPS = 2
    N_STYLE_IMAGES = 20; QUILT_PATCH, QUILT_OVERLAP = 16, 4
    OUT_DIR = "/tmp/PAPER023041_results_test"

# %%
# ===== 2. Google Диск и библиотеки =====
import os, sys, subprocess
if not TEST_MODE:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips", "torchmetrics[image]", "torch-fidelity", "openpyxl", "diffusers>=0.27", "transformers", "accelerate"], check=True)
os.makedirs(f"{OUT_DIR}/images", exist_ok=True); os.makedirs(f"{OUT_DIR}/checkpoints", exist_ok=True)
print("Папка результатов:", OUT_DIR)

# %%
# ===== 3. Код метода, бейзлайнов и метрик =====
import math, time, json, random, glob, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not TEST_MODE: assert DEVICE == "cuda", "GPU не включён! Среда выполнения → Сменить среду выполнения → T4 GPU"
print("Устройство:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
def sync():
    if DEVICE == "cuda": torch.cuda.synchronize()

def set_seed(s):
    random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
def run_seed(tex_id, seed): return seed * 10_000_000 + int(hashlib.md5(tex_id.encode()).hexdigest()[:6], 16)
def gen(s): return torch.Generator(device=DEVICE).manual_seed(int(s))

# ---------- Stage 1 + 2 on GPU: periodic fBM + Voronoi, periodic cubic B-spline warp ----------
def sample_wrap(img, ys, xs):
    H, W = img.shape; y0f, x0f = torch.floor(ys), torch.floor(xs); wy, wx = ys - y0f, xs - x0f
    y0, x0 = y0f.long() % H, x0f.long() % W; y1, x1 = (y0 + 1) % H, (x0 + 1) % W
    return (img[y0, x0] * (1 - wy) * (1 - wx) + img[y0, x1] * (1 - wy) * wx +
            img[y1, x0] * wy * (1 - wx) + img[y1, x1] * wy * wx)

_GRID = {}
def pix_grid(R):
    if R not in _GRID:
        a = torch.arange(R, device=DEVICE, dtype=torch.float32)
        _GRID[R] = torch.meshgrid(a, a, indexing="ij")
    return _GRID[R]

def procedural(R, alpha, g, octaves=6, n_sites=30):
    Y, X = pix_grid(R); f = torch.zeros(R, R, device=DEVICE); amp, s = 1.0, 0.0
    for i in range(octaves):
        res = 2 ** (i + 1); grid = torch.rand(res, res, generator=g, device=DEVICE)
        f += amp * sample_wrap(grid, Y * res / R, X * res / R); s += amp; amp *= 0.5
    f /= s
    sites = torch.rand(n_sites, 2, generator=g, device=DEVICE) * R
    dy = (Y[..., None] - sites[:, 0]).abs(); dx = (X[..., None] - sites[:, 1]).abs()
    dy = torch.minimum(dy, R - dy); dx = torch.minimum(dx, R - dx)
    d = torch.sqrt(dy ** 2 + dx ** 2).min(-1).values; d = d / d.max()
    b = alpha * f + (1 - alpha) * d
    return (b - b.min()) / (b.max() - b.min() + 1e-12)

_BASIS = {}
def bspline_basis(R, n):
    """Periodic uniform cubic B-spline basis N_i(u) evaluated at R texels: matrix R x n (Eq. 4-6)."""
    if (R, n) not in _BASIS:
        u = np.arange(R) * n / R; i = np.floor(u).astype(int); t = u - i
        w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
        B = np.zeros((R, n))
        for k, wk in enumerate(w): np.add.at(B, (np.arange(R), (i - 1 + k) % n), wk)
        _BASIS[(R, n)] = torch.tensor(B, dtype=torch.float32, device=DEVICE)
    return _BASIS[(R, n)]

def bspline_warp(tex, g, grid_n=8, sigma=0.05):
    R = tex.shape[0]; B = bspline_basis(R, grid_n)
    cu = torch.randn(grid_n, grid_n, generator=g, device=DEVICE) * sigma
    cv = torch.randn(grid_n, grid_n, generator=g, device=DEVICE) * sigma
    du, dv = (B @ cu @ B.T) * R, (B @ cv @ B.T) * R    # S(u,v) = sum_ij N_i(u) N_j(v) P_ij
    Y, X = pix_grid(R)
    return sample_wrap(tex, Y + dv, X + du)

def make_content(R, g, alpha=ALPHA, warp=True):
    t = procedural(R, alpha, g)
    return bspline_warp(t, g) if warp else t

# ---------- VGG19 encoder (correct layers) ----------
LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 15: "relu3_3", 20: "relu4_1"}
class VGGEnc(nn.Module):
    def __init__(self):
        super().__init__()
        w = None if TEST_MODE else torchvision.models.VGG19_Weights.IMAGENET1K_V1
        self.vgg = torchvision.models.vgg19(weights=w).features[:21].eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std; out = {}
        for i, l in enumerate(self.vgg):
            x = l(x)
            if i in LAYERS: out[LAYERS[i]] = x
        return out
ENC = VGGEnc().to(DEVICE)
STYLE_L = ("relu1_1", "relu2_1", "relu3_1")

def gram(f):
    b, c, h, w = f.shape; f = f.view(b, c, h * w); return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
def tv_loss(x): return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
def T(a): return a[None, None] if torch.is_tensor(a) else torch.tensor(a, dtype=torch.float32, device=DEVICE)[None, None]

# ---------- Stage 3a: per-texture U-Net optimization (Hybrid-Opt) ----------
def cbr(i, o): return [nn.ReflectionPad2d(1), nn.Conv2d(i, o, 3), nn.InstanceNorm2d(o), nn.ReLU(True)]
class Block(nn.Module):
    def __init__(s, i, o): super().__init__(); s.b = nn.Sequential(*cbr(i, o), *cbr(o, o))
    def forward(s, x): return s.b(x)
class Up(nn.Module):
    def __init__(s, i, o): super().__init__(); s.c = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.ReflectionPad2d(1), nn.Conv2d(i, o, 3))
    def forward(s, x): return s.c(x)
class LightUNet(nn.Module):
    def __init__(s, ch=1, b=24):
        super().__init__()
        s.e1, s.e2, s.e3, s.mid = Block(ch, b), Block(b, 2 * b), Block(2 * b, 4 * b), Block(4 * b, 4 * b)
        s.pool = nn.MaxPool2d(2); s.u2, s.d2 = Up(4 * b, 2 * b), Block(4 * b, 2 * b); s.u1, s.d1 = Up(2 * b, b), Block(2 * b, b)
        s.out = nn.Conv2d(b, ch, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); m = s.mid(s.e3(s.pool(e2)))
        d2 = s.d2(torch.cat([s.u2(m), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return torch.sigmoid(s.out(d1))

def opt_refine(content, ref, lam_s, steps=None):
    steps = OPT_STEPS if steps is None else steps
    model = LightUNet().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1)); x, r = T(content), T(ref)
    with torch.no_grad():
        fs = ENC(r); gs = {k: gram(fs[k]) for k in STYLE_L}; fc = ENC(x)["relu3_3"]
    for _ in range(steps):
        opt.zero_grad(); y = model(x); fy = ENC(y)
        loss = lam_s * sum(F.mse_loss(gram(fy[k]), gs[k]) for k in STYLE_L) + LAMBDA_C * F.mse_loss(fy["relu3_3"], fc) + LAMBDA_TV * tv_loss(y)
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad(): return model(x)[0, 0].clamp(0, 1)

# ---------- Stage 3b: feed-forward AdaIN refinement (Hybrid-FF) ----------
def mstd(f, eps=1e-5):
    b, c = f.shape[:2]; v = f.view(b, c, -1)
    return v.mean(-1).view(b, c, 1, 1), (v.var(-1) + eps).sqrt().view(b, c, 1, 1)
def adain(fc, fs):
    mc, sc = mstd(fc); ms, ss = mstd(fs); return (fc - mc) / sc * ss + ms
def rc(i, o, relu=True): return [nn.ReflectionPad2d(1), nn.Conv2d(i, o, 3)] + ([nn.ReLU(True)] if relu else [])
class Decoder(nn.Module):
    def __init__(s):
        super().__init__(); up = lambda: nn.Upsample(scale_factor=2, mode="nearest")
        s.net = nn.Sequential(*rc(512, 256), up(), *rc(256, 256), *rc(256, 256), *rc(256, 256), *rc(256, 128), up(),
                              *rc(128, 128), *rc(128, 64), up(), *rc(64, 64), *rc(64, 1, relu=False))
    def forward(s, x): return s.net(x)

def ff_refine(dec, content, ref, a):
    fc = ENC(T(content))["relu4_1"]; fs = ENC(T(ref))["relu4_1"]
    t = a * adain(fc, fs) + (1 - a) * fc
    return dec(t)[0, 0].clamp(0, 1)

# ---------- Baseline: VGG19 Gram optimization (correct layers relu1_1..relu4_1) ----------
def vgg_gram_opt(ref, steps, g):
    L = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")
    with torch.no_grad(): fs = ENC(T(ref)); gs = {k: gram(fs[k]) for k in L}
    x = torch.rand(1, 1, *ref.shape, generator=g, device=DEVICE).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.02)
    for _ in range(steps):
        opt.zero_grad(); fy = ENC(x.clamp(0, 1)); loss = 1e5 * sum(F.mse_loss(gram(fy[k]), gs[k]) for k in L)
        loss.backward(); opt.step()
    return x.detach().clamp(0, 1)[0, 0]

# ---------- Baseline: Efros-Freeman quilting (unchanged) ----------
def _min_cut_v(err):
    H, W = err.shape; E = err.copy(); back = np.zeros((H, W), int)
    for i in range(1, H):
        for j in range(W):
            lo, hi = max(j - 1, 0), min(j + 2, W); k = lo + int(np.argmin(E[i - 1, lo:hi])); back[i, j] = k; E[i, j] += E[i - 1, k]
    mask = np.ones((H, W)); j = int(np.argmin(E[-1]))
    for i in range(H - 1, -1, -1): mask[i, :j] = 0; j = back[i, j]
    return mask
def quilting(ref, out_size, rng, patch, ov, n_cand=600, tol=0.1):
    H, W = ref.shape; step = patch - ov; n = math.ceil((out_size - ov) / step); S = n * step + ov
    out = np.zeros((S, S), np.float32); win = np.lib.stride_tricks.sliding_window_view(ref, (patch, patch))
    for bi in range(n):
        for bj in range(n):
            y, x = bi * step, bj * step
            cand = win[rng.integers(0, H - patch + 1, n_cand), rng.integers(0, W - patch + 1, n_cand)].astype(np.float32)
            if bi == 0 and bj == 0: out[:patch, :patch] = cand[0]; continue
            err = np.zeros(n_cand, np.float32); cur = out[y:y + patch, x:x + patch]
            if bj > 0: err += ((cand[:, :, :ov] - cur[None, :, :ov]) ** 2).sum((1, 2))
            if bi > 0: err += ((cand[:, :ov, :] - cur[None, :ov, :]) ** 2).sum((1, 2))
            p = cand[rng.choice(np.where(err <= err.min() * (1 + tol) + 1e-8)[0])]; mask = np.ones((patch, patch))
            if bj > 0: mask[:, :ov] = np.minimum(mask[:, :ov], _min_cut_v((p[:, :ov] - cur[:, :ov]) ** 2))
            if bi > 0: mask[:ov, :] = np.minimum(mask[:ov, :], _min_cut_v(((p[:ov, :] - cur[:ov, :]) ** 2).T).T)
            out[y:y + patch, x:x + patch] = mask * p + (1 - mask) * cur
    return out[:out_size, :out_size]

# ---------- Metrics ----------
import lpips
LPIPS_NET = lpips.LPIPS(net="alex", pretrained=not TEST_MODE, pnet_rand=TEST_MODE, verbose=False).to(DEVICE)
def metrics(ref, out):
    with torch.no_grad():
        l = LPIPS_NET(T(ref).repeat(1, 3, 1, 1) * 2 - 1, T(out).repeat(1, 3, 1, 1) * 2 - 1).item()
    return structural_similarity(ref, out, data_range=1.0), peak_signal_noise_ratio(ref, out, data_range=1.0), l
print("Код загружен.")

# %%
# ===== 4. Датасет DTD: тест, валидация и обучающие изображения (без пересечений) =====
def load_gray(path, R):
    im = Image.open(path).convert("L"); w, h = im.size; m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((R, R), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0
if TEST_MODE:
    DTD_DIR = "/tmp/fake_dtd/images"
    if not os.path.isdir(DTD_DIR):
        rr = np.random.default_rng(0)
        for c in CATEGORIES + ["dotted", "lined"]:
            os.makedirs(f"{DTD_DIR}/{c}", exist_ok=True)
            for k in range(8):
                yy, xx = np.mgrid[0:150, 0:170]; a = np.sin(xx / (2 + k) + rr.random() * 6) * np.cos(yy / (3 + k))
                Image.fromarray(((a + 1) * 127).astype(np.uint8)).save(f"{DTD_DIR}/{c}/{c}_{k:04d}.jpg")
else:
    DTD_DIR = "/content/dtd/images"
    if not os.path.isdir(DTD_DIR):
        subprocess.run("wget -q https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -O /content/dtd.tar.gz && tar -xzf /content/dtd.tar.gz -C /content", shell=True, check=True)
TEST_SET, VAL_SET = [], []
for c in CATEGORIES:
    files = sorted(glob.glob(f"{DTD_DIR}/{c}/*.jpg"))
    TEST_SET += [(os.path.basename(f), c, f) for f in files[:N_TEST_PER_CAT]]
    VAL_SET += [(os.path.basename(f), c, f) for f in files[N_TEST_PER_CAT:N_TEST_PER_CAT + N_VAL_PER_CAT]]
HIGH_SET = [t for c in CATEGORIES for t in [x for x in TEST_SET if x[1] == c][:N_512_PER_CAT]]
EXCL = {p for _, _, p in TEST_SET + VAL_SET}
pool = sorted(p for p in glob.glob(f"{DTD_DIR}/*/*.jpg") if p not in EXCL)
TRAIN_FILES = sorted(np.random.default_rng(0).choice(pool, min(N_STYLE_IMAGES, len(pool)), replace=False).tolist())
assert not (set(TRAIN_FILES) & EXCL), "пересечение обучающих и тестовых текстур!"
print(f"Тест {len(TEST_SET)}, 512-набор {len(HIGH_SET)}, валидация {len(VAL_SET)}, обучение {len(TRAIN_FILES)} (без пересечений)")
pd.DataFrame({"train_file": [os.path.relpath(p, DTD_DIR) for p in TRAIN_FILES]}).to_csv(f"{OUT_DIR}/ff_training_files.csv", index=False)

# %%
# ===== 5. Проверка входных данных и загрузка обученных сетей =====
RAW3 = f"{OUT_DIR}/raw_results.csv"
assert os.path.exists(RAW3), "Не найден raw_results.csv версии 3 в папке PAPER023041_results на Google Диске!"
raw3 = pd.read_csv(RAW3)
if not TEST_MODE:
    assert (raw3.get("version", pd.Series(dtype=str)) == "v3").sum() >= 2000, "raw_results.csv не содержит результатов версии 3 — откройте ноутбук в том аккаунте, где выполнялась версия 3.1"
    for w in FF_STYLE_W_GRID: assert os.path.exists(f"{OUT_DIR}/checkpoints/ff_w{w:g}.pt"), f"Нет обученной сети ff_w{w:g}.pt в checkpoints"
def load_short(path, short=288):
    im = Image.open(path).convert("L"); w, h = im.size; k = short / min(w, h)
    return np.asarray(im.resize((max(FF_CROP, round(w * k)), max(FF_CROP, round(h * k))), Image.BICUBIC), np.uint8)
_POOL = []
def style_batch(rng):
    if not _POOL: _POOL.extend(load_short(p, max(288, FF_CROP)) for p in TRAIN_FILES)
    out = []
    for _ in range(FF_BATCH):
        im = _POOL[rng.integers(len(_POOL))]; y = rng.integers(0, im.shape[0] - FF_CROP + 1); x = rng.integers(0, im.shape[1] - FF_CROP + 1)
        out.append(im[y:y + FF_CROP, x:x + FF_CROP])
    return torch.tensor(np.stack(out), dtype=torch.float32, device=DEVICE)[:, None] / 255.0

def train_ff(style_w, content="procedural"):
    tag = f"ff_w{style_w:g}" if content == "procedural" else f"ff_noise_w{style_w:g}"
    ck = f"{OUT_DIR}/checkpoints/{tag}.pt"
    dec = Decoder().to(DEVICE); opt = torch.optim.Adam(dec.parameters(), lr=FF_LR); it0, secs = 0, 0.0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEVICE, weights_only=True); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"]); it0, secs = st["it"], st["secs"]
    if it0 >= FF_ITERS: return dec.eval(), secs
    set_seed(12345 + it0); rng = np.random.default_rng(12345 + it0); g = gen(777 + it0); t0 = time.time()
    for it in range(it0, FF_ITERS):
        s = style_batch(rng)
        if content == "procedural":
            c = torch.stack([make_content(FF_CROP, g, alpha=float(rng.random()), warp=bool(rng.random() < 0.5)) for _ in range(FF_BATCH)])[:, None]
        else:  # i.i.d. uniform white noise
            c = torch.rand(FF_BATCH, 1, FF_CROP, FF_CROP, generator=g, device=DEVICE)
        with torch.no_grad(): fc = ENC(c)["relu4_1"]; fs = ENC(s); t = adain(fc, fs["relu4_1"])
        out = dec(t); fo = ENC(out); lc = F.mse_loss(fo["relu4_1"], t); ls = 0.0
        for k in ("relu1_1", "relu2_1", "relu3_1", "relu4_1"):
            mo, so = mstd(fo[k]); mt, st_ = mstd(fs[k]); ls = ls + F.mse_loss(mo, mt) + F.mse_loss(so, st_)
        loss = lc + style_w * ls; opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 250 == 0 or it + 1 == FF_ITERS:
            sync(); el = secs + time.time() - t0
            torch.save({"dec": dec.state_dict(), "opt": opt.state_dict(), "it": it + 1, "secs": el}, ck)
            print(f"  {tag}: итерация {it + 1}/{FF_ITERS}, loss {loss.item():.3f}, {el / 60:.1f} мин")
    sync(); return dec.eval(), secs + time.time() - t0

DECODERS, TRAIN_SECS = {}, {}
for w in FF_STYLE_W_GRID + [FF_W_NEW]:
    DECODERS[w], TRAIN_SECS[w] = train_ff(w)
print("Сети с процедурным входом готовы:", sorted(DECODERS))

# %%
# ===== 6. Выбор (вес стиля, α) на валидации для предлагаемого метода БЕЗ B-сплайна =====
SENS4 = f"{OUT_DIR}/sensitivity_v4.csv"
sens4 = pd.read_csv(SENS4) if os.path.exists(SENS4) else pd.DataFrame()
def done4(param, v): return len(sens4) and ((sens4.param == param) & np.isclose(sens4.value, v)).sum() >= len(VAL_SET)
def val_eval(param, v, fn, grad=False):
    global sens4
    if done4(param, v): return
    rows = []
    for tex_id, cat, path in VAL_SET:
        ref = load_gray(path, RES_MAIN); s = run_seed(tex_id, 0); set_seed(s); g = gen(s)
        sync(); t0 = time.perf_counter()
        with (torch.enable_grad() if grad else torch.no_grad()): out = fn(ref, g, tex_id, cat, s)
        sync(); ms = (time.perf_counter() - t0) * 1000
        o = out.detach().cpu().numpy() if torch.is_tensor(out) else out
        sm, ps, lp = metrics(ref, o); rows.append(dict(param=param, value=float(v), texture_id=tex_id, category=cat, ssim=sm, psnr=ps, lpips=lp, time_ms=ms))
    sens4 = pd.concat([sens4, pd.DataFrame(rows)], ignore_index=True); sens4.to_csv(SENS4, index=False); print(f"  {param} = {v:g}: готово")
def best4(param): return float(sens4[sens4.param == param].groupby("value").lpips.mean().idxmin())
for w in sorted(DECODERS):
    val_eval("nowarp_style_w", w, lambda ref, g, *a, w=w: ff_refine(DECODERS[w], make_content(RES_MAIN, g, warp=False), ref, 1.0))
W_SEL = best4("nowarp_style_w")
for a in FF_ALPHA_GRID:
    val_eval("nowarp_alpha", a, lambda ref, g, *x, a=a: ff_refine(DECODERS[W_SEL], make_content(RES_MAIN, g, warp=False), ref, a))
A_SEL = best4("nowarp_alpha"); DEC = DECODERS[W_SEL]
print(f"Выбрано на валидации (без сплайна): вес стиля = {W_SEL:g}, α = {A_SEL:g}")
if W_SEL == min(DECODERS): print("ВНИМАНИЕ: лучший вес стиля на границе сетки — сообщите Claude.")

# обучение сети на белом шуме с тем же весом стиля
DEC_NOISE, NOISE_SECS = train_ff(W_SEL, content="noise")
print(f"Сеть на белом шуме готова ({NOISE_SECS / 60:.1f} мин)")

# %%
# ===== 7. Stable Diffusion v1.5 + IP-Adapter (видит образец) =====
_IP = None
def load_ip():
    global _IP
    if _IP is None and not TEST_MODE:
        from diffusers import StableDiffusionPipeline
        _IP = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
        _IP.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        _IP.set_progress_bar_config(disable=True)
    return _IP
IP_PROMPT = "seamless texture, top-down photo, high detail"
def sd_ip(ref, out_size, seed, scale):
    if TEST_MODE: return np.random.default_rng(seed).random((out_size, out_size)).astype(np.float32)
    pipe = load_ip(); pipe.set_ip_adapter_scale(scale)
    ex = Image.fromarray((ref * 255).astype(np.uint8)).convert("RGB").resize((512, 512), Image.BICUBIC)
    g = torch.Generator(DEVICE).manual_seed(int(seed) % (2**31))
    img = pipe(IP_PROMPT, ip_adapter_image=ex, num_inference_steps=SD_STEPS, height=512, width=512, generator=g).images[0]
    return np.asarray(img.convert("L").resize((out_size, out_size), Image.BICUBIC), np.float32) / 255.0
for sc in IP_SCALES:
    val_eval("ip_scale", sc, lambda ref, g, tex, cat, s, sc=sc: sd_ip(ref, RES_MAIN, s, sc))
IP_SEL = best4("ip_scale"); print(f"Выбрано на валидации: IP-Adapter scale = {IP_SEL:g}")

# %%
# ===== 8. Тестовые запуски новых вариантов =====
from tqdm.auto import tqdm
RAW4 = f"{OUT_DIR}/raw_results_v4.csv"
P4 = "Proposed: fBM + FF (no B-spline)"
NEW = {P4: [RES_MAIN, RES_HIGH], "White noise + FF": [RES_MAIN, RES_HIGH], "SD v1.5 + IP-Adapter": [RES_MAIN, RES_HIGH]}
def run4(m, ref, tex_id, cat, seed):
    s = run_seed(tex_id, seed); set_seed(s); g = gen(s); R = ref.shape[0]
    sync()
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0; t0 = time.perf_counter()
    with torch.no_grad():
        if m == P4: out = ff_refine(DEC, make_content(R, g, warp=False), ref, A_SEL)
        elif m == "White noise + FF": out = ff_refine(DEC_NOISE, torch.rand(R, R, generator=g, device=DEVICE), ref, A_SEL)
        elif m == "SD v1.5 + IP-Adapter": out = sd_ip(ref, R, s, IP_SEL)
        else: raise ValueError(m)
    sync(); ms = (time.perf_counter() - t0) * 1000
    mem = (torch.cuda.max_memory_allocated() - m0) / 2**20 if DEVICE == "cuda" else 0.0
    out = out.detach().cpu().numpy() if torch.is_tensor(out) else out
    return out.astype(np.float32), ms, mem
def fname(R, m, sd, tex_id):
    return f"{R}_{re.sub(r'[^A-Za-z0-9]+', '_', m).strip('_')}_{sd}_{tex_id.replace('.jpg', '.png')}"
import re
raw4 = pd.read_csv(RAW4) if os.path.exists(RAW4) else pd.DataFrame(columns=["texture_id", "resolution", "method", "seed"])
# если выбран тот же вариант, что уже посчитан в v3 (вес 3, α=1), строки для 256 берём из v3 (код идентичен)
REUSE_P4 = (W_SEL == 3.0 and A_SEL == 1.0)
if REUSE_P4 and not ((raw4.method == P4) & (raw4.resolution == RES_MAIN)).any() and not TEST_MODE:
    r = raw3[(raw3.method == "fBM + FF (no warp)") & (raw3.resolution == RES_MAIN)].copy(); r["method"] = P4; r["version"] = "v3 (same code, w=3, α=1)"
    raw4 = pd.concat([raw4, r], ignore_index=True); print(f"Строки предлагаемого метода для 256 взяты из v3: {len(r)}")
done = set(zip(raw4.texture_id, raw4.resolution, raw4.method, raw4.seed))
jobs = [(t, R, m) for m, Rs in NEW.items() for R in Rs for t in (TEST_SET if R == RES_MAIN else HIGH_SET)]
todo = [(t, R, m, sd) for (t, R, m) in jobs for sd in SEEDS if (t[0], R, m, sd) not in done]
print("Осталось запусков:", len(todo))
for m in sorted({m for _, _, m, _ in todo}):
    for R in sorted({R for _, R, mm, _ in todo if mm == m}): run4(m, load_gray(TEST_SET[0][2], R), "warmup", TEST_SET[0][1], 0)
buf, last = [], time.time()
for (tex_id, cat, path), R, m, sd in tqdm(todo):
    ref = load_gray(path, R); out, ms, mem = run4(m, ref, tex_id, cat, sd); s_, p_, l_ = metrics(ref, out)
    fn = fname(R, m, sd, tex_id); Image.fromarray((out * 255).round().astype(np.uint8)).save(f"{OUT_DIR}/images/{fn}")
    buf.append(dict(texture_id=tex_id, category=cat, resolution=R, seed=sd, method=m, ssim=s_, psnr=p_, lpips=l_, time_ms=ms, gpu_mem_mb=mem, image=fn, version="v4"))
    if time.time() - last > 60 or len(buf) >= 50:
        raw4 = pd.concat([raw4, pd.DataFrame(buf)], ignore_index=True); raw4.to_csv(RAW4, index=False); buf, last = [], time.time()
if buf: raw4 = pd.concat([raw4, pd.DataFrame(buf)], ignore_index=True); raw4.to_csv(RAW4, index=False)
print("Тестовые запуски завершены.")

# %%
# ===== 9. FID/KID, разнообразие (LPIPS между сидами), статистика, Excel =====
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from scipy.stats import wilcoxon
import itertools
def u8(a): return torch.tensor((np.stack(a) * 255).round().astype(np.uint8))[:, None].repeat(1, 3, 1, 1)
dist_rows = []
for R, tset in [(RES_MAIN, TEST_SET), (RES_HIGH, HIGH_SET)]:
    real = u8([load_gray(p, R) for _, _, p in tset])
    for m in NEW:
        sub_ = raw4[(raw4.resolution == R) & (raw4.method == m)]
        files = [f"{OUT_DIR}/images/{f}" for f in sub_.image]
        if not len(files) or not all(os.path.exists(f) for f in files):
            dist_rows.append(dict(resolution=R, method=m, note="images missing (reused from v3: use FID of 'fBM + FF (no warp)')")); continue
        fake = u8([np.asarray(Image.open(f), np.float32) / 255 for f in files])
        fid = FrechetInceptionDistance(feature=2048).to(DEVICE); fid.update(real.to(DEVICE), real=True); fid.update(fake.to(DEVICE), real=False)
        kid = KernelInceptionDistance(subset_size=min(50, len(real), len(fake))).to(DEVICE); kid.update(real.to(DEVICE), real=True); kid.update(fake.to(DEVICE), real=False)
        km, ks = kid.compute(); dist_rows.append(dict(resolution=R, method=m, n_real=len(real), n_fake=len(fake), FID=fid.compute().item(), KID_mean=km.item(), KID_std=ks.item()))
        print(R, m, round(dist_rows[-1]["FID"], 1))
dist4 = pd.DataFrame(dist_rows)

# diversity: mean pairwise LPIPS between the 3 seeds of each texture (256)
allraw = pd.concat([raw3, raw4], ignore_index=True)
DIV_M = ["Procedural", "fBM + B-spline", "fBM + FF (no warp)", "Hybrid-FF (Proposed)", "fBM + Opt (no warp)", "Hybrid-Opt",
         "Image-Based", "VGG19 Gram-matrix Opt.", "White noise + FF", "SD v1.5 + IP-Adapter"]
drows = []
for m in DIV_M:
    sub_ = allraw[(allraw.method == m) & (allraw.resolution == RES_MAIN)]
    for tex_id, gdf in sub_.groupby("texture_id"):
        fs = [f"{OUT_DIR}/images/{f}" for f in gdf.sort_values("seed").image]
        if len(fs) < 2 or not all(os.path.exists(f) for f in fs): continue
        ims = [np.asarray(Image.open(f), np.float32) / 255 for f in fs]
        with torch.no_grad():
            d = [LPIPS_NET(T(ims[a]).repeat(1, 3, 1, 1) * 2 - 1, T(ims[b]).repeat(1, 3, 1, 1) * 2 - 1).item() for a, b in itertools.combinations(range(len(ims)), 2)]
        drows.append(dict(method=m, texture_id=tex_id, diversity_lpips=float(np.mean(d))))
div = pd.DataFrame(drows, columns=["method", "texture_id", "diversity_lpips"])
div_sum = div.groupby("method").diversity_lpips.agg(["count", "mean", "std"]).reset_index()
print(div_sum.round(4).to_string(index=False))

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r_, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r_) * ps[i])); adj[i] = run
    return adj
MET = ["ssim", "psnr", "lpips", "time_ms"]
pt = allraw.groupby(["resolution", "method", "texture_id"], as_index=False)[MET + ["gpu_mem_mb"]].mean()
summary4 = pt[pt.method.isin(list(NEW))].groupby(["resolution", "method"]).agg(n=("texture_id", "count"),
    **{f"{k}_{f}": (k, f) for k in MET + ["gpu_mem_mb"] for f in ("mean", "std")}).reset_index()
BASE = ["Procedural", "Image-Based", "VGG19 Gram-matrix Opt.", "Stable Diffusion", "Hybrid-Opt", "Hybrid-FF (Proposed)", "White noise + FF", "SD v1.5 + IP-Adapter"]
wrows = []
for R in (RES_MAIN, RES_HIGH):
    for b in BASE:
        A = pt[(pt.resolution == R) & (pt.method == P4)].set_index("texture_id"); B = pt[(pt.resolution == R) & (pt.method == b)].set_index("texture_id")
        idx = A.index.intersection(B.index)
        if len(idx) < 5: continue
        for k in MET:
            try: st, p = wilcoxon(A.loc[idx, k], B.loc[idx, k])
            except ValueError: st, p = np.nan, np.nan
            wrows.append(dict(resolution=R, method_A=P4, method_B=b, metric=k, n_pairs=len(idx), median_diff=float(np.median(A.loc[idx, k] - B.loc[idx, k])), W=st, p=p))
for a_, b_ in [("fBM + B-spline", "Procedural"), ("Hybrid-FF (Proposed)", "fBM + FF (no warp)"), ("Hybrid-Opt", "fBM + Opt (no warp)"), (P4, "White noise + FF")]:
    A = div[div.method == a_].set_index("texture_id").diversity_lpips; B = div[div.method == b_].set_index("texture_id").diversity_lpips
    idx = A.index.intersection(B.index)
    if len(idx) >= 5:
        st, p = wilcoxon(A.loc[idx], B.loc[idx]); wrows.append(dict(resolution=RES_MAIN, method_A=a_, method_B=b_, metric="diversity_lpips", n_pairs=len(idx), median_diff=float(np.median(A.loc[idx] - B.loc[idx])), W=st, p=p))
wil4 = pd.DataFrame(wrows); wil4["p_holm"] = np.nan
for R in wil4.resolution.unique():
    msk = (wil4.resolution == R) & wil4.p.notna(); wil4.loc[msk, "p_holm"] = holm(wil4.loc[msk, "p"])
sens_sum4 = sens4.groupby(["param", "value"]).agg(n=("texture_id", "count"), ssim=("ssim", "mean"), psnr=("psnr", "mean"), lpips=("lpips", "mean"), time_ms=("time_ms", "mean")).reset_index()
cfg4 = pd.DataFrame([("selected_style_w_nowarp", W_SEL), ("selected_alpha_nowarp", A_SEL), ("selected_ip_scale", IP_SEL), ("reuse_256_rows_from_v3", REUSE_P4),
    ("train_minutes", json.dumps({str(k): round(v / 60, 1) for k, v in TRAIN_SECS.items()})), ("noise_decoder_minutes", round(NOISE_SECS / 60, 1)),
    ("ip_adapter", "h94/IP-Adapter ip-adapter_sd15.bin, prompt: " + IP_PROMPT), ("GPU", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"),
    ("torch", torch.__version__), ("test_mode", TEST_MODE)], columns=["key", "value"])
XL = f"{OUT_DIR}/RESULTS_PAPER023041_v4.xlsx"
with pd.ExcelWriter(XL) as w:
    summary4.to_excel(w, sheet_name="summary_new", index=False); wil4.to_excel(w, sheet_name="wilcoxon", index=False)
    dist4.to_excel(w, sheet_name="fid_kid_new", index=False); div_sum.to_excel(w, sheet_name="diversity_summary", index=False)
    div.to_excel(w, sheet_name="diversity_raw", index=False); sens_sum4.to_excel(w, sheet_name="validation_summary", index=False)
    sens4.to_excel(w, sheet_name="validation_raw", index=False); raw4.to_excel(w, sheet_name="raw_v4", index=False); cfg4.to_excel(w, sheet_name="config", index=False)
print("ГОТОВО! Файл:", XL)
print(summary4[["resolution", "method", "lpips_mean", "time_ms_mean"]].round(3).to_string(index=False))
