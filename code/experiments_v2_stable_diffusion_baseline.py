# %% [markdown]
# # PAPER023041 — повторный запуск экспериментов (ревизия для IJIGSP)
#
# **Как запустить (ничего менять не нужно):**
# 1. Вверху: **Среда выполнения → Сменить среду выполнения → T4 GPU → Сохранить**.
# 2. **Среда выполнения → Выполнить все** (Ctrl+F9).
# 3. Появится окно доступа к Google Диску — разрешите (результаты сохраняются туда).
# 4. Ждите. Весь прогон занимает примерно **4–7 часов**. Вкладку держите открытой.
# 5. Если Colab отключился — просто снова **Выполнить все**. Уже посчитанное не пересчитывается, работа продолжится с места остановки.
# 6. В конце на Google Диске в папке **PAPER023041_results** появится файл **RESULTS_PAPER023041.xlsx**. Его и пришлите.
#
# Что исправлено по сравнению со старым кодом:
# * CNN-уточнение теперь использует текстуру DTD как стилевой образец (раньше — собственный вход);
# * время меряется **полностью**, включая 150 шагов оптимизации (одинаково для всех методов, с `cuda.synchronize`);
# * фиксируются все сиды (Python, NumPy, PyTorch, CUDA, cuDNN);
# * Wilcoxon считается по 75 текстурам (сначала усреднение по 3 сидам), с поправкой Holm;
# * λ_s выбирается на **отдельных** валидационных текстурах, которые не входят в тестовый набор;
# * добавлен KID (надёжнее FID при малом числе изображений).

# %%
# ===== 1. НАСТРОЙКИ (менять не нужно) =====
TEST_MODE = False          # True — только для проверки кода без GPU
CATEGORIES = ["woven", "braided", "stratified", "knitted", "marbled"]
N_TEST_PER_CAT = 15        # 15 x 5 = 75 текстур (Table 2, Table 4)
N_512_PER_CAT = 6          # 6 x 5 = 30 текстур для 512x512 (Table 3)
N_VAL_PER_CAT = 3          # 3 x 5 = 15 отдельных текстур для подбора гиперпараметров
SEEDS = (0, 1, 2)
RES_MAIN, RES_HIGH = 256, 512
ALPHA = 0.5                # смешивание fBM/Voronoi (Eq. 3)
CNN_STEPS = 150
LAMBDA_S_DEFAULT, LAMBDA_C, LAMBDA_TV = 100.0, 0.5, 1e-4
VGG_GRAM_STEPS = 250
SD_STEPS = 25
QUILT_PATCH, QUILT_OVERLAP = 32, 8
SENS_GRID = {"lambda_s": [10, 100, 1e3, 1e4, 1e5, 1e6, 1e7],
             "lambda_c": [0.1, 0.5, 1.0, 2.0],
             "lambda_tv": [0.0, 1e-5, 1e-4, 1e-3],
             "steps": [50, 100, 150, 250, 500]}
OUT_DIR = "/content/drive/MyDrive/PAPER023041_results"

if TEST_MODE:
    N_TEST_PER_CAT, N_512_PER_CAT, N_VAL_PER_CAT = 2, 1, 1
    SEEDS = (0, 1)
    RES_MAIN, RES_HIGH = 64, 96
    CNN_STEPS, VGG_GRAM_STEPS, SD_STEPS = 3, 3, 2
    QUILT_PATCH, QUILT_OVERLAP = 16, 4
    SENS_GRID = {"lambda_s": [10, 100], "lambda_c": [0.5], "lambda_tv": [1e-4], "steps": [2, 3]}
    OUT_DIR = "/tmp/PAPER023041_results_test"

# %%
# ===== 2. Подключение Google Диска и установка библиотек =====
import os, sys, subprocess
if not TEST_MODE:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips", "torchmetrics[image]",
                    "torch-fidelity", "diffusers>=0.27", "transformers", "accelerate", "openpyxl"], check=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
print("Папка результатов:", OUT_DIR)

# %%
# ===== 3. Общий код: процедурная генерация, B-сплайн, U-Net, бейзлайны, метрики =====
import math, time, json, random, glob, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not TEST_MODE:
    assert DEVICE == "cuda", "GPU не включён! Среда выполнения → Сменить среду выполнения → T4 GPU"
print("Устройство:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_seed(tex_id, seed):
    """Детерминированный сид для пары (текстура, сид) — одинаковый при повторном запуске."""
    h = int(hashlib.md5(tex_id.encode()).hexdigest()[:6], 16)
    return seed * 10_000_000 + h

# ---------- Stage 1: fBM + Voronoi (Eq. 2-3) ----------
def _value_noise(shape, res, rng):
    grid = rng.random((res + 1, res + 1))
    x = np.linspace(0, res, shape[0], endpoint=False); y = np.linspace(0, res, shape[1], endpoint=False)
    xi, yi = x.astype(int), y.astype(int); xf, yf = x - xi, y - yi
    top = grid[xi][:, yi] * (1 - xf)[:, None] + grid[xi + 1][:, yi] * xf[:, None]
    bot = grid[xi][:, yi + 1] * (1 - xf)[:, None] + grid[xi + 1][:, yi + 1] * xf[:, None]
    return top * (1 - yf)[None, :] + bot * yf[None, :]

def generate_base_texture(shape, alpha, rng, octaves=6, n_sites=30):
    f = np.zeros(shape); amp, s = 1.0, 0.0
    for i in range(octaves):
        f += amp * _value_noise(shape, 2 ** (i + 1), rng); s += amp; amp *= 0.5
    f /= s
    sites = rng.random((n_sites, 2)) * np.array(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    c = np.stack([xx, yy], -1).reshape(-1, 2)
    d = np.sqrt(((c[:, None, :] - sites[None]) ** 2).sum(-1)).min(1).reshape(shape); d /= d.max()
    b = alpha * f + (1 - alpha) * d
    return ((b - b.min()) / (b.max() - b.min() + 1e-12)).astype(np.float32)

# ---------- Stage 2: B-spline structural warp (Eq. 4-7) ----------
def bspline_warp(tex, rng, grid_n=8, sigma=0.05):
    H, W = tex.shape
    cu, cv = rng.normal(0, sigma, (grid_n, grid_n)), rng.normal(0, sigma, (grid_n, grid_n))
    cx, cy = np.linspace(0, H - 1, grid_n), np.linspace(0, W - 1, grid_n)
    du = RectBivariateSpline(cx, cy, cu, kx=3, ky=3)(np.arange(H), np.arange(W)) * H
    dv = RectBivariateSpline(cx, cy, cv, kx=3, ky=3)(np.arange(H), np.arange(W)) * W
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([np.clip(yy + dv, 0, H - 1), np.clip(xx + du, 0, W - 1)])
    return map_coordinates(tex, coords, order=1, mode="reflect").astype(np.float32)

# ---------- VGG19 features (shared by CNN refinement and VGG19 Gram baseline) ----------
class VGGFeatures(nn.Module):
    IDX = {3: "relu1_1", 8: "relu2_1", 13: "relu3_1", 15: "relu3_3", 22: "relu4_1"}
    def __init__(self):
        super().__init__()
        w = None if TEST_MODE else torchvision.models.VGG19_Weights.IMAGENET1K_V1
        self.vgg = torchvision.models.vgg19(weights=w).features[:23].eval()
        for p in self.vgg.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std; out = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.IDX: out[self.IDX[i]] = x
        return out

def gram(f):
    b, c, h, w = f.shape; f = f.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

def tv_loss(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()

# ---------- Stage 3: lightweight U-Net (Eq. 8) ----------
class ConvBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o), nn.ReLU(True),
                            nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o), nn.ReLU(True))
    def forward(s, x): return s.b(x)

class LightUNet(nn.Module):
    def __init__(s, ch=1, base=24):
        super().__init__()
        s.e1, s.e2, s.e3 = ConvBlock(ch, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        s.pool, s.mid = nn.MaxPool2d(2), ConvBlock(base * 4, base * 4)
        s.u2, s.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), ConvBlock(base * 4, base * 2)
        s.u1, s.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), ConvBlock(base * 2, base)
        s.out = nn.Conv2d(base, ch, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2)); m = s.mid(e3)
        d2 = s.d2(torch.cat([s.u2(m), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return torch.sigmoid(s.out(d1))

def to_t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)[None, None]

def cnn_refine(warped, ref, vgg, steps, lam_s, lam_c, lam_tv):
    """Per-texture fine-tuning: style target = DTD reference (FIXED), content target = warped input."""
    model = LightUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    x, r = to_t(warped), to_t(ref)
    with torch.no_grad():
        fs = vgg(r); gs = {k: gram(fs[k]) for k in ("relu1_1", "relu2_1", "relu3_1")}
        fc = vgg(x)["relu3_3"]
    for _ in range(steps):
        opt.zero_grad(); y = model(x); fy = vgg(y)
        ls = sum(F.mse_loss(gram(fy[k]), gs[k]) for k in gs)
        lc = F.mse_loss(fy["relu3_3"], fc)
        loss = lam_s * ls + lam_c * lc + lam_tv * tv_loss(y)
        loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        return model(x)[0, 0].clamp(0, 1).cpu().numpy()

# ---------- Baseline: VGG19 Gram-matrix optimization (Gatys et al.) ----------
def vgg_gram_opt(ref, vgg, steps, rng):
    layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")
    with torch.no_grad():
        fs = vgg(to_t(ref)); gs = {k: gram(fs[k]) for k in layers}
    x = to_t(rng.random(ref.shape).astype(np.float32)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.02)
    for _ in range(steps):
        opt.zero_grad(); fy = vgg(x.clamp(0, 1))
        loss = 1e5 * sum(F.mse_loss(gram(fy[k]), gs[k]) for k in layers)
        loss.backward(); opt.step()
    return x.detach().clamp(0, 1)[0, 0].cpu().numpy()

# ---------- Baseline: Efros-Freeman image quilting ----------
def _min_cut_v(err):  # err: H x ov, returns mask (1 = take new patch)
    H, W = err.shape; E = err.copy(); back = np.zeros((H, W), int)
    for i in range(1, H):
        for j in range(W):
            lo, hi = max(j - 1, 0), min(j + 2, W)
            k = lo + int(np.argmin(E[i - 1, lo:hi])); back[i, j] = k; E[i, j] += E[i - 1, k]
    mask = np.ones((H, W)); j = int(np.argmin(E[-1]))
    for i in range(H - 1, -1, -1):
        mask[i, :j] = 0; j = back[i, j]
    return mask

def quilting(ref, out_size, rng, patch, ov, n_cand=600, tol=0.1):
    H, W = ref.shape; step = patch - ov; n = math.ceil((out_size - ov) / step)
    S = n * step + ov; out = np.zeros((S, S), np.float32)
    win = np.lib.stride_tricks.sliding_window_view(ref, (patch, patch))
    for bi in range(n):
        for bj in range(n):
            y, x = bi * step, bj * step
            ys = rng.integers(0, H - patch + 1, n_cand); xs = rng.integers(0, W - patch + 1, n_cand)
            cand = win[ys, xs].astype(np.float32)
            if bi == 0 and bj == 0:
                out[:patch, :patch] = cand[0]; continue
            err = np.zeros(n_cand, np.float32); cur = out[y:y + patch, x:x + patch]
            if bj > 0: err += ((cand[:, :, :ov] - cur[None, :, :ov]) ** 2).sum((1, 2))
            if bi > 0: err += ((cand[:, :ov, :] - cur[None, :ov, :]) ** 2).sum((1, 2))
            ok = np.where(err <= err.min() * (1 + tol) + 1e-8)[0]; p = cand[rng.choice(ok)]
            mask = np.ones((patch, patch))
            if bj > 0: mask[:, :ov] = np.minimum(mask[:, :ov], _min_cut_v((p[:, :ov] - cur[:, :ov]) ** 2))
            if bi > 0: mask[:ov, :] = np.minimum(mask[:ov, :], _min_cut_v(((p[:ov, :] - cur[:ov, :]) ** 2).T).T)
            out[y:y + patch, x:x + patch] = mask * p + (1 - mask) * cur
    return out[:out_size, :out_size]

# ---------- Baseline: Stable Diffusion v1.5 ----------
_SD = None
def load_sd():
    global _SD
    if _SD is None and not TEST_MODE:
        from diffusers import StableDiffusionPipeline
        _SD = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                      torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
        _SD.set_progress_bar_config(disable=True)
    return _SD

def sd_texture(category, out_size, seed):
    if TEST_MODE:
        return np.random.default_rng(seed).random((out_size, out_size)).astype(np.float32)
    g = torch.Generator(DEVICE).manual_seed(seed % (2**31))
    img = load_sd()(f"seamless {category} texture, top-down photo, high detail", num_inference_steps=SD_STEPS,
                    height=512, width=512, generator=g).images[0]
    img = img.convert("L").resize((out_size, out_size), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.0

# ---------- Metrics ----------
import lpips
LPIPS_NET = lpips.LPIPS(net="alex", pretrained=not TEST_MODE, pnet_rand=TEST_MODE, verbose=False).to(DEVICE)
def metrics(ref, out):
    s = structural_similarity(ref, out, data_range=1.0)
    p = peak_signal_noise_ratio(ref, out, data_range=1.0)
    with torch.no_grad():
        l = LPIPS_NET(to_t(ref).repeat(1, 3, 1, 1) * 2 - 1, to_t(out).repeat(1, 3, 1, 1) * 2 - 1).item()
    return s, p, l

VGG = VGGFeatures().to(DEVICE)
print("Код загружен. Параметров U-Net:", sum(p.numel() for p in LightUNet().parameters()))

# ---------- One method call, timed end-to-end ----------
def run_method(method, ref, tex_id, category, seed, lam_s=None, lam_c=LAMBDA_C, lam_tv=LAMBDA_TV, steps=None):
    lam_s = LAMBDA_S if lam_s is None else lam_s; steps = CNN_STEPS if steps is None else steps
    s = run_seed(tex_id, seed); set_seed(s); rng = np.random.default_rng(s); R = ref.shape[0]
    if DEVICE == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0
    t0 = time.perf_counter()
    if method == "Procedural":
        out = generate_base_texture((R, R), ALPHA, rng)
    elif method == "fBM + B-spline":
        out = bspline_warp(generate_base_texture((R, R), ALPHA, rng), rng)
    elif method == "fBM + CNN":
        out = cnn_refine(generate_base_texture((R, R), ALPHA, rng), ref, VGG, steps, lam_s, lam_c, lam_tv)
    elif method == "Hybrid (Proposed)":
        out = cnn_refine(bspline_warp(generate_base_texture((R, R), ALPHA, rng), rng), ref, VGG, steps, lam_s, lam_c, lam_tv)
    elif method == "Image-Based":
        out = quilting(ref, R, rng, QUILT_PATCH, QUILT_OVERLAP)
    elif method == "VGG19 Gram-matrix Opt.":
        out = vgg_gram_opt(ref, VGG, VGG_GRAM_STEPS, rng)
    elif method == "Stable Diffusion":
        out = sd_texture(category, R, s)
    else:
        raise ValueError(method)
    if DEVICE == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    mem = (torch.cuda.max_memory_allocated() - base_mem) / 2**20 if DEVICE == "cuda" else 0.0
    return out.astype(np.float32), ms, mem

# %%
# ===== 4. Загрузка датасета DTD =====
def load_gray(path, R):
    im = Image.open(path).convert("L"); w, h = im.size; m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((R, R), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0

if TEST_MODE:
    DTD_DIR = "/tmp/fake_dtd/images"
    for c in CATEGORIES:
        os.makedirs(f"{DTD_DIR}/{c}", exist_ok=True)
        for k in range(N_TEST_PER_CAT + N_VAL_PER_CAT):
            rr = np.random.default_rng(hash((c, k)) % 2**32)
            Image.fromarray((generate_base_texture((120, 140), rr.random(), rr) * 255).astype(np.uint8)).save(f"{DTD_DIR}/{c}/{c}_{k:04d}.jpg")
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
print(f"Тест: {len(TEST_SET)} текстур, 512-набор: {len(HIGH_SET)}, валидация: {len(VAL_SET)}")
pd.DataFrame(TEST_SET, columns=["texture_id", "category", "path"]).drop(columns="path").to_csv(f"{OUT_DIR}/texture_list.csv", index=False)

# %%
# ===== 5. Подбор гиперпараметров на валидационных текстурах (Minor 2) =====
# Этап 1: перебор λ_s (остальное по умолчанию) -> выбор λ_s по минимальному LPIPS.
# Этап 2: перебор λ_c, λ_tv и числа шагов уже при выбранном λ_s.
def sweep(csv, param, values, base_cfg):
    df = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    if len(df) and "param" in df:
        df = df[df.param.isin(SENS_GRID.keys())]
    done = set(zip(df.get("param", []), map(float, df.get("value", [])), df.get("texture_id", [])))
    for v in values:
        rows = []
        for tex_id, cat, path in VAL_SET:
            if (param, float(v), tex_id) in done: continue
            cfg = dict(base_cfg); cfg[param] = v
            ref = load_gray(path, RES_MAIN)
            out, ms, mem = run_method("Hybrid (Proposed)", ref, tex_id, cat, 0, lam_s=cfg["lambda_s"],
                                      lam_c=cfg["lambda_c"], lam_tv=cfg["lambda_tv"], steps=int(cfg["steps"]))
            sm, ps, lp = metrics(ref, out)
            rows.append(dict(param=param, value=float(v), base_lambda_s=cfg["lambda_s"], texture_id=tex_id,
                             category=cat, ssim=sm, psnr=ps, lpips=lp, time_ms=ms))
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True); df.to_csv(csv, index=False)
        print(f"  {param} = {v}: готово")
    return df

defaults = dict(lambda_s=LAMBDA_S_DEFAULT, lambda_c=LAMBDA_C, lambda_tv=LAMBDA_TV, steps=CNN_STEPS)
SENS1_CSV = f"{OUT_DIR}/sensitivity.csv"            # этап 1 (строки lambda_s)
sens1 = sweep(SENS1_CSV, "lambda_s", SENS_GRID["lambda_s"], defaults)
sens1 = sens1[sens1.param == "lambda_s"]
ls = sens1.groupby("value").lpips.mean()
LAMBDA_S = float(ls.idxmin())
print("LPIPS по λ_s:\n", ls.round(4).to_string(), f"\n→ выбрано λ_s = {LAMBDA_S}")
if LAMBDA_S == max(SENS_GRID["lambda_s"]):
    print("ВНИМАНИЕ: лучший λ_s на границе сетки — сообщите об этом Claude.")

SENS2_CSV = f"{OUT_DIR}/sensitivity_stage2.csv"     # этап 2 (при выбранном λ_s)
stage2 = dict(defaults, lambda_s=LAMBDA_S)
sens2 = pd.DataFrame()
for param in ("lambda_c", "lambda_tv", "steps"):
    sens2 = sweep(SENS2_CSV, param, SENS_GRID[param], stage2)
sens2 = sens2[sens2.base_lambda_s == LAMBDA_S] if len(sens2) else sens2
sens = pd.concat([sens1.assign(stage=1), sens2.assign(stage=2)], ignore_index=True)
json.dump({"lambda_s_selected": LAMBDA_S}, open(f"{OUT_DIR}/selected.json", "w"))

# %%
# ===== 6. Основные эксперименты: Table 2 (256), Table 4 (абляция), Table 3 (512) =====
from tqdm.auto import tqdm
RAW_CSV = f"{OUT_DIR}/raw_results.csv"
MAIN_METHODS = ["Procedural", "Image-Based", "VGG19 Gram-matrix Opt.", "Stable Diffusion", "Hybrid (Proposed)"]
ABL_METHODS = ["fBM + B-spline", "fBM + CNN"]
jobs = [(t, RES_MAIN, m) for t in TEST_SET for m in MAIN_METHODS + ABL_METHODS] + \
       [(t, RES_HIGH, m) for t in HIGH_SET for m in MAIN_METHODS]
raw = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else pd.DataFrame()
CNN_METHODS = {"Hybrid (Proposed)", "fBM + CNN"}
if len(raw):  # строки CNN-методов, посчитанные с другим λ_s, пересчитываются
    if "lambda_s" not in raw: raw["lambda_s"] = np.nan
    stale = raw.method.isin(CNN_METHODS) & (raw.lambda_s != LAMBDA_S)
    if stale.any(): print(f"Пересчитаю {int(stale.sum())} строк CNN-методов (другой λ_s)")
    raw = raw[~stale].reset_index(drop=True)
done = set(zip(raw.get("texture_id", []), raw.get("resolution", []), raw.get("method", []), raw.get("seed", [])))
todo = [(t, R, m, sd) for (t, R, m) in jobs for sd in SEEDS if (t[0], R, m, sd) not in done]
print(f"Всего запусков: {len(jobs) * len(SEEDS)}, осталось: {len(todo)}")
if any(m == "Stable Diffusion" for _, _, m, _ in todo): load_sd()
buf, last = [], time.time()
for (tex_id, cat, path), R, m, sd in tqdm(todo):
    ref = load_gray(path, R)
    out, ms, mem = run_method(m, ref, tex_id, cat, sd)
    s, p, l = metrics(ref, out)
    fn = f"{OUT_DIR}/images/{R}_{m.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p').replace('.', '')}_{sd}_{tex_id.replace('.jpg', '.png')}"
    Image.fromarray((out * 255).round().astype(np.uint8)).save(fn)
    buf.append(dict(texture_id=tex_id, category=cat, resolution=R, seed=sd, method=m, ssim=s, psnr=p, lpips=l,
                    time_ms=ms, gpu_mem_mb=mem, image=os.path.basename(fn),
                    lambda_s=LAMBDA_S if m in CNN_METHODS else np.nan))
    if time.time() - last > 60 or len(buf) >= 50:
        raw = pd.concat([raw, pd.DataFrame(buf)], ignore_index=True); raw.to_csv(RAW_CSV, index=False); buf, last = [], time.time()
if buf:
    raw = pd.concat([raw, pd.DataFrame(buf)], ignore_index=True); raw.to_csv(RAW_CSV, index=False)
print("Основные эксперименты завершены.")

# %%
# ===== 7. FID и KID по методам =====
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
def u8(a): return torch.tensor((np.stack(a) * 255).round().astype(np.uint8))[:, None].repeat(1, 3, 1, 1)
dist_rows = []
for R, tset, methods in [(RES_MAIN, TEST_SET, MAIN_METHODS + ABL_METHODS), (RES_HIGH, HIGH_SET, MAIN_METHODS)]:
    real = u8([load_gray(p, R) for _, _, p in tset])
    for m in methods:
        sub = raw[(raw.resolution == R) & (raw.method == m)]
        fake = u8([np.asarray(Image.open(f"{OUT_DIR}/images/{f}"), np.float32) / 255 for f in sub.image])
        try:
            fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
            fid.update(real.to(DEVICE), real=True); fid.update(fake.to(DEVICE), real=False)
            kid = KernelInceptionDistance(subset_size=min(50, len(real), len(fake))).to(DEVICE)
            kid.update(real.to(DEVICE), real=True); kid.update(fake.to(DEVICE), real=False)
            km, ks = kid.compute()
            dist_rows.append(dict(resolution=R, method=m, n_real=len(real), n_fake=len(fake), FID=fid.compute().item(),
                                  KID_mean=km.item(), KID_std=ks.item()))
        except Exception as e:
            dist_rows.append(dict(resolution=R, method=m, n_real=len(real), n_fake=len(fake), error=str(e)[:200]))
        print(R, m, dist_rows[-1].get("FID"), dist_rows[-1].get("KID_mean"))
dist = pd.DataFrame(dist_rows)

# %%
# ===== 8. Статистика: средние, Wilcoxon (75 пар, поправка Holm) и сохранение в Excel =====
from scipy.stats import wilcoxon
MET = ["ssim", "psnr", "lpips", "time_ms"]
per_tex = raw.groupby(["resolution", "method", "texture_id", "category"], as_index=False)[MET + ["gpu_mem_mb"]].mean()
summary = per_tex.groupby(["resolution", "method"]).agg(
    n=("texture_id", "count"), **{f"{k}_{f}": (k, f) for k in MET + ["gpu_mem_mb"] for f in ("mean", "std")}).reset_index()

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

comps = [(RES_MAIN, "Table 2", "Hybrid (Proposed)", b) for b in MAIN_METHODS[:-1]] + \
        [(RES_MAIN, "Table 4", a, "Procedural") for a in ABL_METHODS + ["Hybrid (Proposed)"]] + \
        [(RES_MAIN, "Table 4", "Hybrid (Proposed)", "fBM + CNN")] + \
        [(RES_HIGH, "Table 3", "Hybrid (Proposed)", b) for b in MAIN_METHODS[:-1]]
wrows = []
for R, tab, a, b in comps:
    A = per_tex[(per_tex.resolution == R) & (per_tex.method == a)].set_index("texture_id")
    B = per_tex[(per_tex.resolution == R) & (per_tex.method == b)].set_index("texture_id")
    idx = A.index.intersection(B.index)
    for k in MET:
        d = A.loc[idx, k] - B.loc[idx, k]
        try: st, p = wilcoxon(A.loc[idx, k], B.loc[idx, k], alternative="two-sided")
        except ValueError: st, p = np.nan, np.nan
        wrows.append(dict(table=tab, resolution=R, method_A=a, method_B=b, metric=k, n_pairs=len(idx),
                          median_diff_A_minus_B=float(np.median(d)) if len(d) else np.nan, W=st, p=p))
wil = pd.DataFrame(wrows)
wil["p_holm"] = np.nan
for tab in wil.table.unique():
    msk = (wil.table == tab) & wil.p.notna(); wil.loc[msk, "p_holm"] = holm(wil.loc[msk, "p"])
wil["significant_0.05_holm"] = wil.p_holm < 0.05
wil["H0"] = "median of paired differences = 0"; wil["H1"] = "median of paired differences != 0 (two-sided)"

sens_sum = sens.groupby(["stage", "param", "value"]).agg(n=("texture_id", "count"), ssim=("ssim", "mean"), psnr=("psnr", "mean"),
                                                 lpips=("lpips", "mean"), time_ms=("time_ms", "mean")).reset_index()
import platform, scipy, skimage
cfg = pd.DataFrame(list({
    "GPU": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU", "torch": torch.__version__,
    "torchvision": torchvision.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__,
    "python": platform.python_version(), "seeds": str(SEEDS), "seed_rule": "seed*1e7 + md5(texture_id); random/numpy/torch/cuda seeded, cudnn deterministic",
    "lambda_s_selected_on_validation": LAMBDA_S, "lambda_c": LAMBDA_C, "lambda_tv": LAMBDA_TV, "cnn_steps": CNN_STEPS,
    "vgg_gram_steps": VGG_GRAM_STEPS, "sd_model": "stable-diffusion-v1-5, fp16, 512px, resized", "sd_steps": SD_STEPS,
    "quilting": f"patch {QUILT_PATCH}, overlap {QUILT_OVERLAP}", "alpha": ALPHA, "color": "grayscale (1 channel)",
    "timing": "end-to-end per texture incl. all optimization steps, cuda.synchronize, models preloaded",
    "metrics_reference": "SSIM/PSNR/LPIPS: output vs. the DTD reference crop of the same texture (not pixel-aligned)",
    "test_mode": TEST_MODE}.items()), columns=["key", "value"])

XL = f"{OUT_DIR}/RESULTS_PAPER023041.xlsx"
with pd.ExcelWriter(XL) as w:
    summary.to_excel(w, sheet_name="summary", index=False); wil.to_excel(w, sheet_name="wilcoxon", index=False)
    dist.to_excel(w, sheet_name="fid_kid", index=False); sens_sum.to_excel(w, sheet_name="sensitivity_summary", index=False)
    sens.to_excel(w, sheet_name="sensitivity_raw", index=False); raw.to_excel(w, sheet_name="raw", index=False)
    per_tex.to_excel(w, sheet_name="per_texture_mean", index=False); cfg.to_excel(w, sheet_name="config", index=False)
print("ГОТОВО! Файл:", XL)

# %%
# ===== 9. Рисунок-пример для статьи (300 dpi) =====
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = MAIN_METHODS
fig, ax = plt.subplots(len(CATEGORIES), len(show) + 1, figsize=(1.6 * (len(show) + 1), 1.6 * len(CATEGORIES)))
for i, c in enumerate(CATEGORIES):
    tex_id, _, path = [t for t in TEST_SET if t[1] == c][0]
    ax[i, 0].imshow(load_gray(path, RES_MAIN), cmap="gray", vmin=0, vmax=1)
    for j, m in enumerate(show):
        f = raw[(raw.texture_id == tex_id) & (raw.method == m) & (raw.resolution == RES_MAIN) & (raw.seed == SEEDS[0])].image.iloc[0]
        ax[i, j + 1].imshow(np.asarray(Image.open(f"{OUT_DIR}/images/{f}")), cmap="gray", vmin=0, vmax=255)
for j, t in enumerate(["DTD reference"] + show):
    ax[0, j].set_title(t, fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/Fig_examples_300dpi.png", dpi=300); plt.close()
print("Рисунок сохранён: Fig_examples_300dpi.png")
