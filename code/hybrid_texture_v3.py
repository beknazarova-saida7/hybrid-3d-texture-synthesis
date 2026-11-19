# %% [markdown]
# # PAPER023041 — эксперименты, версия 3.1 (улучшенный метод)
#
# **Как запустить:** Среда выполнения → Сменить среду выполнения → **T4 GPU** → Сохранить, затем **Среда выполнения → Выполнить все**. Разрешите доступ к Google Диску.
# Время: примерно **4–5 часов** (из них ~1 час — однократное обучение сети). Если Colab отключился — снова «Выполнить все», работа продолжится.
# **Перед запуском:** на Google Диске должна быть папка **PAPER023041_results** с файлом **raw_results.csv** (из архива, который прислал Claude). Из него берутся результаты Stable Diffusion прошлого прогона (код не менялся). Квилтинг пересчитывается заново (быстро). Всё остальное тоже пересчитывается.
# В конце пришлите **RESULTS_PAPER023041_v3.xlsx** и **Fig_examples_v3_300dpi.png** из папки PAPER023041_results.
#
# Изменения по сравнению с v2 (все будут честно описаны в статье и в ответе рецензентам):
# 1. Исправлены слои VGG19: relu1_1, relu2_1, relu3_1, relu4_1 (раньше по ошибке relu1_2, relu2_2, relu3_2, relu4_2).
# 2. Процедурная текстура, Voronoi и B-сплайн — периодические (бесшовные), без обрезки координат на краях; B-сплайн — точное тензорное произведение кубических базисов (Eq. 4); всё на GPU.
# 3. U-Net: reflect-padding и upsample+conv вместо ConvTranspose (убирает «шахматку» и полосы по краям).
# 4. Новый вариант этапа 3 — **Hybrid-FF**: сеть прямого прохода (AdaIN, Huang & Belongie 2017), обучается **один раз** на изображениях DTD, которые **не входят** в тестовый и валидационный наборы. Вариант с оптимизацией на каждую текстуру (**Hybrid-Opt**) тоже оценивается.
# 5. Все гиперпараметры (вес стиля при обучении, α смешивания, λ_s) выбираются только на 15 валидационных текстурах.
# 6. Прогрев GPU перед замером времени для всех методов.

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
FF_ALPHA_GRID = [0.25, 0.5, 0.75, 1.0]
N_STYLE_IMAGES = 2500
QUILT_PATCH, QUILT_OVERLAP = 32, 8
OUT_DIR = "/content/drive/MyDrive/PAPER023041_results"
if TEST_MODE:
    N_TEST_PER_CAT, N_512_PER_CAT, N_VAL_PER_CAT = 2, 1, 1
    SEEDS = (0, 1); RES_MAIN, RES_HIGH = 64, 96
    OPT_STEPS, VGG_GRAM_STEPS = 3, 3; OPT_LAMBDA_S_GRID = [1e3, 1e5]
    FF_ITERS, FF_BATCH, FF_CROP = 3, 2, 64; FF_STYLE_W_GRID = [3.0, 10.0]; FF_ALPHA_GRID = [0.5, 1.0]
    N_STYLE_IMAGES = 20; QUILT_PATCH, QUILT_OVERLAP = 16, 4
    OUT_DIR = "/tmp/PAPER023041_results_test"

# %%
# ===== 2. Google Диск и библиотеки =====
import os, sys, subprocess
if not TEST_MODE:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips", "torchmetrics[image]", "torch-fidelity", "openpyxl"], check=True)
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
# ===== 5. Однократное обучение сети Hybrid-FF (для каждого веса стиля из сетки) =====
def load_short(path, short=288):
    im = Image.open(path).convert("L"); w, h = im.size; k = short / min(w, h)
    return np.asarray(im.resize((max(FF_CROP, round(w * k)), max(FF_CROP, round(h * k))), Image.BICUBIC), np.uint8)
STYLE_POOL = [load_short(p, max(288, FF_CROP)) for p in TRAIN_FILES]
def style_batch(rng):
    out = []
    for _ in range(FF_BATCH):
        im = STYLE_POOL[rng.integers(len(STYLE_POOL))]; y = rng.integers(0, im.shape[0] - FF_CROP + 1); x = rng.integers(0, im.shape[1] - FF_CROP + 1)
        out.append(im[y:y + FF_CROP, x:x + FF_CROP])
    return torch.tensor(np.stack(out), dtype=torch.float32, device=DEVICE)[:, None] / 255.0

def train_ff(style_w):
    ck = f"{OUT_DIR}/checkpoints/ff_w{style_w:g}.pt"
    dec = Decoder().to(DEVICE); opt = torch.optim.Adam(dec.parameters(), lr=FF_LR); it0, secs = 0, 0.0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEVICE); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"]); it0, secs = st["it"], st["secs"]
    if it0 >= FF_ITERS: return dec.eval(), secs
    set_seed(12345 + it0); rng = np.random.default_rng(12345 + it0); g = gen(777 + it0); t0 = time.time()
    for it in range(it0, FF_ITERS):
        s = style_batch(rng)
        c = torch.stack([make_content(FF_CROP, g, alpha=float(rng.random()), warp=bool(rng.random() < 0.5)) for _ in range(FF_BATCH)])[:, None]
        with torch.no_grad(): fc = ENC(c)["relu4_1"]; fs = ENC(s); t = adain(fc, fs["relu4_1"])
        out = dec(t); fo = ENC(out)
        lc = F.mse_loss(fo["relu4_1"], t)
        ls = 0.0
        for k in ("relu1_1", "relu2_1", "relu3_1", "relu4_1"):
            mo, so = mstd(fo[k]); mt, st_ = mstd(fs[k]); ls = ls + F.mse_loss(mo, mt) + F.mse_loss(so, st_)
        loss = lc + style_w * ls; opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 250 == 0 or it + 1 == FF_ITERS:
            sync(); el = secs + time.time() - t0
            torch.save({"dec": dec.state_dict(), "opt": opt.state_dict(), "it": it + 1, "secs": el}, ck)
            print(f"  w={style_w:g}: итерация {it + 1}/{FF_ITERS}, loss {loss.item():.3f}, {el / 60:.1f} мин")
    sync(); return dec.eval(), secs + time.time() - t0

DECODERS, TRAIN_SECS = {}, {}
for w in FF_STYLE_W_GRID:
    DECODERS[w], TRAIN_SECS[w] = train_ff(w)
print("Обучение завершено:", {k: f"{v / 60:.1f} мин" for k, v in TRAIN_SECS.items()})

# %%
# ===== 6. Подбор гиперпараметров на валидационных текстурах =====
SENS_CSV = f"{OUT_DIR}/sensitivity_v3.csv"
sens = pd.read_csv(SENS_CSV) if os.path.exists(SENS_CSV) else pd.DataFrame()
def have(param, v): return len(sens) and ((sens.param == param) & np.isclose(sens.value, v)).sum() >= len(VAL_SET)
def val_eval(param, v, fn):
    global sens
    if have(param, v): return
    rows = []
    for tex_id, cat, path in VAL_SET:
        ref = load_gray(path, RES_MAIN); s = run_seed(tex_id, 0); set_seed(s); g = gen(s)
        sync(); t0 = time.perf_counter()
        with torch.no_grad() if param != "opt_lambda_s" else torch.enable_grad(): out = fn(ref, g)
        sync(); ms = (time.perf_counter() - t0) * 1000
        o = out.cpu().numpy() if torch.is_tensor(out) else out
        sm, ps, lp = metrics(ref, o); rows.append(dict(param=param, value=float(v), texture_id=tex_id, category=cat, ssim=sm, psnr=ps, lpips=lp, time_ms=ms))
    sens = pd.concat([sens, pd.DataFrame(rows)], ignore_index=True); sens.to_csv(SENS_CSV, index=False); print(f"  {param} = {v:g}: готово")
def best(param): return float(sens[sens.param == param].groupby("value").lpips.mean().idxmin())

for w in FF_STYLE_W_GRID:
    val_eval("ff_style_w", w, lambda ref, g, w=w: ff_refine(DECODERS[w], make_content(RES_MAIN, g), ref, 1.0))
FF_W = best("ff_style_w")
for a in FF_ALPHA_GRID:
    val_eval("ff_alpha", a, lambda ref, g, a=a: ff_refine(DECODERS[FF_W], make_content(RES_MAIN, g), ref, a))
FF_A = best("ff_alpha")
for l in OPT_LAMBDA_S_GRID:
    val_eval("opt_lambda_s", l, lambda ref, g, l=l: opt_refine(make_content(RES_MAIN, g), ref, l))
OPT_LS = best("opt_lambda_s")
DEC = DECODERS[FF_W]
print(f"Выбрано на валидации: вес стиля = {FF_W:g}, α = {FF_A:g}, λ_s (Opt) = {OPT_LS:g}")
for name, grid, v in [("вес стиля", FF_STYLE_W_GRID, FF_W), ("α", FF_ALPHA_GRID, FF_A), ("λ_s", OPT_LAMBDA_S_GRID, OPT_LS)]:
    if v == max(grid) and len(grid) > 1: print(f"ВНИМАНИЕ: лучший {name} на границе сетки — сообщите Claude.")

# %%
# ===== 7. Основные эксперименты =====
from tqdm.auto import tqdm
RAW_CSV = f"{OUT_DIR}/raw_results.csv"
REUSED = {"Image-Based", "Stable Diffusion"}           # код не менялся — берём из прошлого прогона
NEW_MAIN = ["Procedural", "VGG19 Gram-matrix Opt.", "Hybrid-Opt", "Hybrid-FF (Proposed)"]
ABL = ["fBM + B-spline", "fBM + FF (no warp)", "fBM + Opt (no warp)"]
MAIN_METHODS = ["Procedural", "Image-Based", "VGG19 Gram-matrix Opt.", "Stable Diffusion", "Hybrid-Opt", "Hybrid-FF (Proposed)"]

def run_method(m, ref, tex_id, cat, seed):
    s = run_seed(tex_id, seed); set_seed(s); g = gen(s); rng = np.random.default_rng(s); R = ref.shape[0]
    sync()
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0; t0 = time.perf_counter()
    if m == "Procedural":
        with torch.no_grad(): out = make_content(R, g, warp=False)
    elif m == "fBM + B-spline":
        with torch.no_grad(): out = make_content(R, g, warp=True)
    elif m == "Hybrid-FF (Proposed)":
        with torch.no_grad(): out = ff_refine(DEC, make_content(R, g, warp=True), ref, FF_A)
    elif m == "fBM + FF (no warp)":
        with torch.no_grad(): out = ff_refine(DEC, make_content(R, g, warp=False), ref, FF_A)
    elif m == "Hybrid-Opt":
        out = opt_refine(make_content(R, g, warp=True).detach(), ref, OPT_LS)
    elif m == "fBM + Opt (no warp)":
        out = opt_refine(make_content(R, g, warp=False).detach(), ref, OPT_LS)
    elif m == "VGG19 Gram-matrix Opt.":
        out = vgg_gram_opt(ref, VGG_GRAM_STEPS, g)
    elif m == "Image-Based":
        out = quilting(ref, R, rng, QUILT_PATCH, QUILT_OVERLAP)
    else: raise ValueError(m)
    sync(); ms = (time.perf_counter() - t0) * 1000
    mem = (torch.cuda.max_memory_allocated() - m0) / 2**20 if DEVICE == "cuda" else 0.0
    out = out.detach().cpu().numpy() if torch.is_tensor(out) else out
    return out.astype(np.float32), ms, mem

jobs = [(t, RES_MAIN, m) for t in TEST_SET for m in NEW_MAIN + ABL + ["Image-Based"]] + [(t, RES_HIGH, m) for t in HIGH_SET for m in NEW_MAIN + ["Image-Based"]]
raw = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else pd.DataFrame(columns=["texture_id", "resolution", "method", "seed"])
if "version" not in raw: raw["version"] = "v2"
raw = raw[raw.method.isin(REUSED) | (raw.version == "v3")].reset_index(drop=True)
if "image" not in raw: raw["image"] = ""
img_ok = raw.image.fillna("").map(lambda f: bool(f) and os.path.exists(f"{OUT_DIR}/images/{f}"))
# квилтинг без сохранённых картинок пересчитываем (детерминирован, результат совпадёт с v2)
raw = raw[~((raw.method == "Image-Based") & ~img_ok & (raw.version != "v3"))].reset_index(drop=True)
img_ok = raw.image.fillna("").map(lambda f: bool(f) and os.path.exists(f"{OUT_DIR}/images/{f}"))
SD_IMAGES_MISSING = bool(((raw.method == "Stable Diffusion") & ~img_ok).any())
n_sd = int((raw.method == "Stable Diffusion").sum())
print(f"Stable Diffusion из прошлого прогона: {n_sd} строк" + (" (картинок нет — FID/KID возьмём из прошлого прогона)" if SD_IMAGES_MISSING else ""))
if not TEST_MODE: assert n_sd == 315, "Не найден raw_results.csv с результатами Stable Diffusion в папке PAPER023041_results на Google Диске!"
done = set(zip(raw.texture_id, raw.resolution, raw.method, raw.seed))
todo = [(t, R, m, sd) for (t, R, m) in jobs for sd in SEEDS if (t[0], R, m, sd) not in done]
print(f"Переиспользовано строк: {int(raw.method.isin(REUSED).sum())}; осталось запусков: {len(todo)}")
# прогрев GPU для каждого метода (не записывается)
for m in sorted({m for _, _, m, _ in todo}):
    for R in sorted({R for _, R, mm, _ in todo if mm == m}):
        t = TEST_SET[0]; run_method(m, load_gray(t[2], R), "warmup", t[1], 0)
def fname(R, m, sd, tex_id):
    return f"{R}_{m.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p').replace('.', '')}_{sd}_{tex_id.replace('.jpg', '.png')}"
buf, last = [], time.time()
for (tex_id, cat, path), R, m, sd in tqdm(todo):
    ref = load_gray(path, R); out, ms, mem = run_method(m, ref, tex_id, cat, sd); s, p, l = metrics(ref, out)
    fn = fname(R, m, sd, tex_id); Image.fromarray((out * 255).round().astype(np.uint8)).save(f"{OUT_DIR}/images/{fn}")
    buf.append(dict(texture_id=tex_id, category=cat, resolution=R, seed=sd, method=m, ssim=s, psnr=p, lpips=l, time_ms=ms, gpu_mem_mb=mem, image=fn, version="v3"))
    if time.time() - last > 60 or len(buf) >= 50:
        raw = pd.concat([raw, pd.DataFrame(buf)], ignore_index=True); raw.to_csv(RAW_CSV, index=False); buf, last = [], time.time()
if buf: raw = pd.concat([raw, pd.DataFrame(buf)], ignore_index=True); raw.to_csv(RAW_CSV, index=False)
print("Основные эксперименты завершены.")

# %%
# ===== 8. FID и KID =====
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
def u8(a): return torch.tensor((np.stack(a) * 255).round().astype(np.uint8))[:, None].repeat(1, 3, 1, 1)
dist_rows = []
V2_SD_FID = {256: dict(n_real=75, n_fake=225, FID=251.5447, KID_mean=0.0471, KID_std=0.0074),
             512: dict(n_real=30, n_fake=90, FID=276.4147, KID_mean=0.0372, KID_std=0.0069)}
for R, tset, methods in [(RES_MAIN, TEST_SET, MAIN_METHODS + ABL), (RES_HIGH, HIGH_SET, MAIN_METHODS)]:
    real = u8([load_gray(p, R) for _, _, p in tset])
    for m in methods:
        sub = raw[(raw.resolution == R) & (raw.method == m)]
        if not len(sub): continue
        if m == "Stable Diffusion" and SD_IMAGES_MISSING:
            dist_rows.append(dict(resolution=R, method=m, **V2_SD_FID.get(R, dict(FID=np.nan)), source="v2 run 2026-11-16 (same code, same seeds)")); continue
        fake = u8([np.asarray(Image.open(f"{OUT_DIR}/images/{f}"), np.float32) / 255 for f in sub.image])
        try:
            fid = FrechetInceptionDistance(feature=2048).to(DEVICE); fid.update(real.to(DEVICE), real=True); fid.update(fake.to(DEVICE), real=False)
            kid = KernelInceptionDistance(subset_size=min(50, len(real), len(fake))).to(DEVICE); kid.update(real.to(DEVICE), real=True); kid.update(fake.to(DEVICE), real=False)
            km, ks = kid.compute()
            dist_rows.append(dict(resolution=R, method=m, n_real=len(real), n_fake=len(fake), FID=fid.compute().item(), KID_mean=km.item(), KID_std=ks.item()))
        except Exception as e:
            dist_rows.append(dict(resolution=R, method=m, error=str(e)[:200]))
        print(R, m, dist_rows[-1].get("FID"))
dist = pd.DataFrame(dist_rows)

# %%
# ===== 9. Статистика и сохранение =====
from scipy.stats import wilcoxon
MET = ["ssim", "psnr", "lpips", "time_ms"]
per_tex = raw.groupby(["resolution", "method", "texture_id", "category"], as_index=False)[MET + ["gpu_mem_mb"]].mean()
summary = per_tex.groupby(["resolution", "method"]).agg(n=("texture_id", "count"),
    **{f"{k}_{f}": (k, f) for k in MET + ["gpu_mem_mb"] for f in ("mean", "std", "median")}).reset_index()
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
P = "Hybrid-FF (Proposed)"
comps = [(RES_MAIN, "Table 2", P, b) for b in MAIN_METHODS if b != P] + \
        [(RES_MAIN, "Table 4", "fBM + B-spline", "Procedural"), (RES_MAIN, "Table 4", "fBM + FF (no warp)", "Procedural"),
         (RES_MAIN, "Table 4", P, "Procedural"), (RES_MAIN, "Table 4", P, "fBM + FF (no warp)"),
         (RES_MAIN, "Table 4", "Hybrid-Opt", "fBM + Opt (no warp)")] + \
        [(RES_HIGH, "Table 3", P, b) for b in MAIN_METHODS if b != P]
wrows = []
for R, tab, a, b in comps:
    A = per_tex[(per_tex.resolution == R) & (per_tex.method == a)].set_index("texture_id")
    B = per_tex[(per_tex.resolution == R) & (per_tex.method == b)].set_index("texture_id"); idx = A.index.intersection(B.index)
    for k in MET:
        try: st, p = wilcoxon(A.loc[idx, k], B.loc[idx, k], alternative="two-sided")
        except ValueError: st, p = np.nan, np.nan
        wrows.append(dict(table=tab, resolution=R, method_A=a, method_B=b, metric=k, n_pairs=len(idx),
                          median_diff_A_minus_B=float(np.median(A.loc[idx, k] - B.loc[idx, k])) if len(idx) else np.nan, W=st, p=p))
wil = pd.DataFrame(wrows); wil["p_holm"] = np.nan
for tab in wil.table.unique():
    msk = (wil.table == tab) & wil.p.notna(); wil.loc[msk, "p_holm"] = holm(wil.loc[msk, "p"])
wil["significant_0.05_holm"] = wil.p_holm < 0.05
wil["H0"] = "median of paired differences = 0"; wil["H1"] = "median of paired differences != 0 (two-sided)"
sens_sum = sens.groupby(["param", "value"]).agg(n=("texture_id", "count"), ssim=("ssim", "mean"), psnr=("psnr", "mean"), lpips=("lpips", "mean"), time_ms=("time_ms", "mean")).reset_index()
import platform, scipy, skimage
cfg = pd.DataFrame(list({
    "GPU": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU", "torch": torch.__version__, "torchvision": torchvision.__version__,
    "scipy": scipy.__version__, "skimage": skimage.__version__, "python": platform.python_version(),
    "seeds": str(SEEDS), "seed_rule": "seed*1e7 + md5(texture_id); random/numpy/torch/cuda seeded, cudnn deterministic",
    "vgg_layers": "style relu1_1, relu2_1, relu3_1 (+relu4_1 for FF and Gram baseline); content relu3_3 (Opt) / relu4_1 (FF)",
    "ff_decoder_params": sum(p.numel() for p in Decoder().parameters()), "unet_params": sum(p.numel() for p in LightUNet().parameters()),
    "ff_train_iters": FF_ITERS, "ff_batch": FF_BATCH, "ff_crop": FF_CROP, "ff_lr": FF_LR, "ff_train_images": len(TRAIN_FILES),
    "ff_train_minutes_per_model": json.dumps({str(k): round(v / 60, 1) for k, v in TRAIN_SECS.items()}),
    "ff_style_weight_selected_on_val": FF_W, "ff_alpha_selected_on_val": FF_A, "opt_lambda_s_selected_on_val": OPT_LS,
    "opt_steps": OPT_STEPS, "lambda_c": LAMBDA_C, "lambda_tv": LAMBDA_TV, "vgg_gram_steps": VGG_GRAM_STEPS,
    "reused_from_v2": "Stable Diffusion metrics (code unchanged); FID/KID from v2 if images missing; Image-Based recomputed", "procedural": "periodic fBM (6 oct.) + toroidal Voronoi, alpha=0.5, GPU",
    "bspline": "periodic uniform cubic tensor-product B-spline, 8x8 control grid, sigma=0.05, wrap sampling, GPU",
    "timing": "end-to-end per texture incl. procedural, warp and all optimization steps; GPU warm-up; cuda.synchronize; models preloaded",
    "color": "grayscale", "test_mode": TEST_MODE}.items()), columns=["key", "value"])
XL = f"{OUT_DIR}/RESULTS_PAPER023041_v3.xlsx"
with pd.ExcelWriter(XL) as w:
    summary.to_excel(w, sheet_name="summary", index=False); wil.to_excel(w, sheet_name="wilcoxon", index=False)
    dist.to_excel(w, sheet_name="fid_kid", index=False); sens_sum.to_excel(w, sheet_name="sensitivity_summary", index=False)
    sens.to_excel(w, sheet_name="sensitivity_raw", index=False); raw.to_excel(w, sheet_name="raw", index=False)
    per_tex.to_excel(w, sheet_name="per_texture_mean", index=False); cfg.to_excel(w, sheet_name="config", index=False)
print("ГОТОВО! Файл:", XL)
print(summary[summary.resolution == RES_MAIN][["method", "lpips_mean", "time_ms_mean"]].round(3).to_string(index=False))

# %%
# ===== 10. Рисунок для статьи (300 dpi) =====
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = MAIN_METHODS
fig, ax = plt.subplots(len(CATEGORIES), len(show) + 1, figsize=(1.6 * (len(show) + 1), 1.6 * len(CATEGORIES)))
for i, c in enumerate(CATEGORIES):
    tex_id, _, path = [t for t in TEST_SET if t[1] == c][0]
    ax[i, 0].imshow(load_gray(path, RES_MAIN), cmap="gray", vmin=0, vmax=1)
    for j, m in enumerate(show):
        sub = raw[(raw.texture_id == tex_id) & (raw.method == m) & (raw.resolution == RES_MAIN) & (raw.seed == SEEDS[0])]
        if len(sub) and os.path.exists(f"{OUT_DIR}/images/{sub.image.iloc[0]}"): ax[i, j + 1].imshow(np.asarray(Image.open(f"{OUT_DIR}/images/{sub.image.iloc[0]}")), cmap="gray", vmin=0, vmax=255)
for j, t in enumerate(["DTD reference"] + show): ax[0, j].set_title(t, fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/Fig_examples_v3_300dpi.png", dpi=300); plt.close()
print("Рисунок сохранён: Fig_examples_v3_300dpi.png")
