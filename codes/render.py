"""软件渲染器：透视投影 + z-buffer 光栅化 + 高光 + 轮廓线 + 组合排版"""
import struct
import numpy as np
from PIL import Image


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        arr = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return arr[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def _rot(tris, az, el):
    """az: 绕 Z 方位角; el: 相机仰角(0=水平, 90=正上方)"""
    a, e = np.radians(az), np.radians(90.0 - el)
    x, y, z = tris[..., 0], tris[..., 1], tris[..., 2]
    x1 = x * np.cos(a) - y * np.sin(a)
    y1 = x * np.sin(a) + y * np.cos(a)
    y2 = y1 * np.cos(e) - z * np.sin(e)
    z2 = y1 * np.sin(e) + z * np.cos(e)
    return np.stack([x1, y2, z2], axis=-1)


def render(tris, size=(1200, 900), ss=2, az=32, el=26, dist=7.0,
           bg=(250, 250, 248), base=(233, 232, 227), edge=(116, 124, 134),
           zoom=0.82):
    W, H = size[0] * ss, size[1] * ss
    t = _rot(tris, az, el)

    v1, v2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    nrm = np.cross(v1, v2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1
    nrm /= ln
    nrm[nrm[:, 2] < 0] *= -1

    L = np.array([-0.40, -0.52, 0.76])
    L /= np.linalg.norm(L)
    V = np.array([0.0, 0.0, 1.0])
    Hh = (L + V) / np.linalg.norm(L + V)

    lam = np.clip(nrm @ L, 0, 1)
    spec = np.clip(nrm @ Hh, 0, 1) ** 45
    shade = 0.42 + 0.58 * lam ** 0.85
    spec = spec * 42.0

    R = np.abs(t).max()
    D = dist * R
    per = D / (D - t[:, :, 2])
    xy = t[:, :, :2] * per[:, :, None]

    mn, mx = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
    span = mx - mn
    k = min(W * zoom / span[0], H * zoom / span[1])
    off = (np.array([W, H]) - span * k) / 2
    px = (xy - mn) * k + off
    pz = t[:, :, 2]

    depth = np.full((H, W), -1e9, np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.array(bg, np.float32)
    base_arr = np.array(base, np.float32)

    for i in range(len(t)):
        p = px[i]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())) + 1, W)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        w2 = 1 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue
        zz = w0 * pz[i, 0] + w1 * pz[i, 1] + w2 * pz[i, 2]
        sub = depth[y0:y1, x0:x1]
        m = mask & (zz > sub)
        if not m.any():
            continue
        sub[m] = zz[m]
        img[y0:y1, x0:x1][m] = base_arr * shade[i] + spec[i]

    # 轮廓线（深度梯度）
    d = depth.copy()
    valid = d > -1e8
    fill = d[valid].mean() if valid.any() else 0
    d[~valid] = fill
    gy, gx = np.gradient(d)
    g = np.hypot(gx, gy)
    if (g > 0).any():
        thr = np.percentile(g[g > 0], 97) * 1.4
        em = g > thr
        img[em] = img[em] * 0.35 + np.array(edge, np.float32) * 0.65

    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return out.resize(size, Image.LANCZOS)


if __name__ == "__main__":
    from pathlib import Path
    OUT = Path(__file__).resolve().parent.parent / "output"
    OUT.mkdir(parents=True, exist_ok=True)
    tris = read_stl(str(OUT / "riser.stl"))
    print("triangles:", len(tris))

    views = {
        "iso":   dict(az=32, el=26, dist=7.0),
        "front": dict(az=6, el=13, dist=6.5),
        "back":  dict(az=196, el=17, dist=6.5),
    }
    imgs = {}
    for name, kw in views.items():
        imgs[name] = render(tris, **kw)
        imgs[name].save(str(OUT / f"preview_{name}.png"))

    gap, pad = 16, 24
    w, h = imgs["iso"].size
    sheet = Image.new("RGB", (w * 3 + gap * 2 + pad * 2, h + pad * 2), (250, 250, 248))
    for i, name in enumerate(["iso", "front", "back"]):
        sheet.paste(imgs[name], (pad + i * (w + gap), pad))
    sheet.save(str(OUT / "preview_sheet.png"))
    print("ok")
