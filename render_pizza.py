"""Top-down hero shot of a fresh pizza: rendered procedurally with numpy + PIL.

Deterministic, stdlib+numpy/scipy/PIL only. Output 1024x1024 PNG.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFilter

S = 2048
Y, X = np.mgrid[0:S, 0:S].astype(np.float32)
U = (X - S*0.5)/(S*0.5)
V = (Y - S*0.5)/(S*0.5)
RAD = np.sqrt(U*U + V*V)
ANG = np.arctan2(V, U)
rng = np.random.default_rng(11)


def noise_img(seed, cell, shape=(S, S)):
    r = np.random.default_rng(seed)
    h = max(2, shape[0]//cell)
    n = r.random((h, h)).astype(np.float32)
    im = Image.fromarray((n*255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BICUBIC)
    return np.asarray(im, np.float32)/255.0 - 0.5


def fbm(seed, base=128, octaves=4, shape=(S, S)):
    out = np.zeros(shape, np.float32); amp = 1.0; tot = 0.0
    for i in range(octaves):
        out += amp*noise_img(seed + 13*i, max(4, base//(2**i)), shape)
        tot += amp; amp *= 0.5
    return out/tot


def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a+1e-9), 0, 1)
    return t*t*(3-2*t)


# ------------------------------------------------------------- table
grain = 0.5 + 0.5*np.sin(X*0.010 + fbm(3, 256, 3)*7.0)
bg = np.zeros((S, S, 3), np.float32)
bg[..., 0] = 0.075 + 0.035*grain
bg[..., 1] = 0.048 + 0.022*grain
bg[..., 2] = 0.032 + 0.014*grain
# warm key pool upper-left
key = np.exp(-(((X-S*0.30)**2 + (Y-S*0.22)**2)/(S*0.70)**2))*0.75
bg += key[..., None]*np.array([0.30, 0.17, 0.07], np.float32)

img = bg.copy()

# ------------------------------------------------------------- board
bcx, bcy = S*0.50, S*0.52
brx, bry = S*0.475, S*0.455          # board ellipse (near square board, slight perspective)
bd = np.sqrt(((X-bcx)/brx)**2 + ((Y-bcy)/bry)**2)
board_ang = np.arctan2((Y-bcy)/bry, (X-bcx)/brx)
board_wob = 0.012*np.sin(board_ang*3.0) + 0.008*np.sin(board_ang*7.0+1.2)
board = bd < 1.0 + board_wob
# wood board texture: long grain following the x axis
btex = 0.5+0.5*np.sin((X*0.004 + Y*0.0018) + fbm(23, 512, 3)*8.0)
btex2 = 0.5+0.5*np.sin(Y*0.02 + fbm(29, 256, 2)*5.0)
board_col = np.zeros((S, S, 3), np.float32)
board_col[..., 0] = 0.52 + 0.13*btex + 0.03*btex2
board_col[..., 1] = 0.34 + 0.09*btex + 0.02*btex2
board_col[..., 2] = 0.20 + 0.055*btex + 0.012*btex2
# board lighting: brighter toward upper-left, dark rim
bl = np.clip(1.05 - 0.45*np.sqrt(((X-bcx-brx*0.35)/brx)**2 + ((Y-bcy-bry*0.40)/bry)**2), 0.25, 1.6)
board_col *= bl[..., None]
board_col *= (1.0 - 0.55*smoothstep(0.94, 1.0, bd))[..., None]
img = np.where(board[..., None], board_col, img)
# board drop shadow
bsh = np.exp(-(((X-bcx)/(brx*1.02))**2 + ((Y-bcy-bry*0.03)/(bry*1.02))**2-1.0)**2*3.0)
img *= (1.0 - np.where(board, 0.0, 0.45*np.clip(bsh, 0, 1)))[..., None]

# ------------------------------------------------------------- crust ring
pcr = 0.775                                   # pizza radius in RAD units
crust_wob = 0.020*np.sin(ANG*5.0+0.7) + 0.013*np.sin(ANG*9.0+2.1) + 0.010*np.sin(ANG*14.0)
pz_r = pcr + crust_wob                        # outer edge
in_crust = 0.630 + crust_wob*0.5              # inner edge of the crust ring
inside = RAD < pz_r
pizza_shadow_k = np.where(inside, 0.0, 0.60*np.exp(-((RAD-pz_r)**2)*140.0))
img *= (1.0 - pizza_shadow_k)[..., None]

# sauce / cheese base inside the crust
sauce_brown = np.zeros((S, S, 3), np.float32)
sauce_brown[..., 0] = 0.62; sauce_brown[..., 1] = 0.30; sauce_brown[..., 2] = 0.11

# crust ring color: baked dough with puffy volume
crust_t = np.clip((RAD-in_crust)/(pz_r-in_crust+1e-6), 0, 1)
crust_tex = fbm(41, 96, 4)
puff = crust_t*(1-crust_t)*4.0                # 0 at both edges, 1 in middle => rounded ring
crust_lam = 0.55 + 0.75*puff
crust_col = np.zeros((S, S, 3), np.float32)
crust_col[..., 0] = 0.84; crust_col[..., 1] = 0.53; crust_col[..., 2] = 0.21
crust_col *= crust_lam[..., None]
crust_col *= (0.92 + 0.18*crust_tex[..., None])
bakeband = smoothstep(0.25, 0.62, crust_t)*smoothstep(0.95, 0.78, crust_t)
crust_col[..., 0] *= (1.0 + 0.10*bakeband)
crust_col[..., 1] *= (1.0 - 0.05*bakeband)
flour = smoothstep(0.42, 0.5, gaussian_filter(fbm(91, 200, 2), 2))
crust_col += np.array([0.35,0.30,0.22], np.float32)[None,None,:]*(flour*puff)[...,None]*0.5
# darker edges of the crust (inner + outer)
crust_col *= (1.0 - 0.42*smoothstep(0.72, 1.0, crust_t))[..., None]
crust_col *= (1.0 - 0.22*smoothstep(0.20, 0.0, crust_t))[..., None]
# leopard char spots on the puffy ring
char = smoothstep(0.16, 0.34, fbm(47, 48, 4) + 0.10*np.sin(ANG*9.0))
char *= smoothstep(0.22, 0.55, crust_t)*smoothstep(0.95, 0.6, crust_t)
crust_col *= (1.0 - 0.55*char)[..., None]
crust_col += (np.array([1.0, 0.86, 0.6], np.float32)[None, None, :]*0.10*puff[..., None])

# ------------------------------------------------------------- cheese
cheese_mask = smoothstep(in_crust+0.012, in_crust-0.012, RAD)
# molten cheese covers the inner area; sauce peeks through near the crust
sauce_peek = smoothstep(0.66, 0.60, RAD)*0.0
cheese_tex = gaussian_filter(fbm(53, 64, 3), 6)*1.6
cheese_bub = smoothstep(0.22, 0.50, fbm(59, 32, 3) + 0.05)
cheese_col = np.zeros((S, S, 3), np.float32)
cheese_col[..., 0] = 0.92; cheese_col[..., 1] = 0.68; cheese_col[..., 2] = 0.30
cheese_col *= (0.74 + 0.42*cheese_tex[..., None])
# golden toasted patches
toast = smoothstep(0.30, 0.55, gaussian_filter(fbm(61, 40, 3), 8))
cheese_col[..., 0] *= (1.0 + 0.06*toast)
cheese_col[..., 1] *= (1.0 - 0.24*toast)
cheese_col[..., 2] *= (1.0 - 0.42*toast)
# pooled sauce around the edge of the cheese (thin red border)
sauce_edge = smoothstep(0.60, 0.625, RAD)*smoothstep(0.665, 0.64, RAD)
cheese_col = cheese_col*(1-sauce_edge[..., None]) + np.array([0.62, 0.14, 0.05], np.float32)[None, None, :]*sauce_edge[..., None]*1.0
# highlight sheen on the cheese (melted gloss)
gloss_n = fbm(67, 128, 3)
sheen = np.clip(gaussian_filter(np.clip(gloss_n-0.10, 0, None), 8)*2.4, 0, 1)*smoothstep(0.5,0.1,((U+0.35)**2+(V+0.45)**2)**0.5)*0.7
cheese_col += np.array([1.0, 0.93, 0.72], np.float32)[None, None, :]*(sheen**2)[..., None]*0.26
caramel = smoothstep(0.42, 0.62, gaussian_filter(fbm(95, 48, 3), 10))
cheese_col[..., 0] *= (1.0+0.10*caramel)
cheese_col[..., 1] *= (1.0-0.18*caramel)
cheese_col[..., 2] *= (1.0-0.30*caramel)

# assemble crust + cheese
pizza_area = RAD < pz_r
layer = np.where((RAD < in_crust+0.01)[..., None], cheese_col, crust_col)
# cheese bubbles: small wells with dark rim + bright highlight
for i in range(60):
    bu, bv = rng.uniform(-0.52,0.52), rng.uniform(-0.52,0.52)
    if bu**2+bv**2 > 0.36: continue
    br = rng.uniform(0.012, 0.030)
    bd_ = np.sqrt((U-bu)**2+(V-bv)**2)/br
    well = smoothstep(1.0, 0.72, bd_)
    dark_rim = smoothstep(0.8, 1.0, bd_)*smoothstep(1.35, 1.0, bd_)
    img *= (1.0-0.35*well[...,None]*(RAD<in_crust)[...,None])
    img -= (dark_rim[...,None]*np.array([0.10,0.05,0.01],np.float32)[None,None,:])
    hi = smoothstep(0.45, 0.0, bd_)*0.10
    img += (np.array([1.0,0.9,0.7],np.float32)[None,None,:]*hi[...,None])
img = np.where(pizza_area[..., None], layer, img)
# contact shadow just inside the crust ring (cheese melts under crust)
csh = np.exp(-((RAD-in_crust)**2)*(S*0.0)**2/(0.010**2))
csh = np.exp(-((RAD-in_crust)**2)/(0.016**2))*smoothstep(in_crust-0.01, in_crust+0.01, RAD)
img *= (1.0-0.30*np.clip(csh,0,1)[...,None])

# ------------------------------------------------------------- pepperoni
pepperoni = [(-0.42, -0.30, 0.098), (-0.10, -0.46, 0.088), (0.26, -0.34, 0.100),
             (0.52, -0.05, 0.090), (0.30, 0.24, 0.094), (-0.06, 0.18, 0.102),
             (-0.38, 0.30, 0.086), (-0.58, 0.02, 0.082), (0.02, 0.52, 0.080),
             (0.16, -0.03, 0.070)]
for (pu, pv, pr) in pepperoni:
    pd = np.sqrt((U-pu)**2 + (V-pv)**2)/pr
    pm = pd < 1.0
    # cupped disc: bright center, dark raised rim, specular arc upper-left
    rim = smoothstep(0.72, 1.0, pd)
    center = smoothstep(0.6, 0.0, pd)
    pcol = np.array([0.50, 0.11, 0.05], np.float32)[None, None, :]*(0.78+0.22*center[..., None])
    pcol *= (1.0 - 0.18*smoothstep(0.35, 0.0, pd))[..., None]
    pcol *= (1.0 - 0.55*rim)[..., None]
    # broad soft matte sheen only
    # speckles of fat
    fat = smoothstep(0.32, 0.42, fbm(int(abs(pu*1000))+300, 128, 2))
    pcol += np.array([1.0, 0.88, 0.68], np.float32)[None, None, :]*(fat*pm)[..., None]*0.22
    # shadow under the edge
    edge_sh = np.exp(-((pd-1.02)**2)*22.0)
    img = img*(1-edge_sh[..., None]*0.35)
    img = np.where(pm[..., None], pcol, img)

# ------------------------------------------------------------- basil
def leaf(cx_, cy_, ln, wd, rot, seed):
    ca, sa = np.cos(rot), np.sin(rot)
    xr = (X-cx_)*ca + (Y-cy_)*sa
    yr = -(X-cx_)*sa + (Y-cy_)*ca
    t = xr/ln
    wob = 1.0+0.10*np.sin(xr/(ln)*9.0+rot*7.0)
    env = np.clip(1.0-(yr/(wd*wob*(0.35+0.65*np.cos(np.clip(t,-1,1)*1.35))**1.0))**2, 0, None)
    m = (np.abs(t) < 1.0) & (env > 0)
    lcol = np.array([0.10, 0.24, 0.05], np.float32)[None, None, :]*(0.75+0.5*env[..., None])
    lcol += np.array([0.30, 0.46, 0.13], np.float32)[None, None, :]*((env**3)[..., None])*0.35
    lcol += np.array([0.65, 0.85, 0.45], np.float32)[None, None, :]*(np.exp(-(((xr/ln+0.45)**2+(yr/wd*0.9)**2)/(0.5**2)))[..., None])*0.12
    vein = np.exp(-(yr/(wd*0.05))**2)*0.25
    lcol += np.array([0.75, 0.90, 0.45], np.float32)[None, None, :]*vein[..., None]
    sh = gaussian_filter(m.astype(np.float32), 4)*0.35
    img2 = img*(1-sh[..., None])
    lm = smoothstep(0.85, 1.0, env)
    lcol = lcol*(1.0-0.45*lm[..., None])
    img2 = np.where(m[..., None], lcol, img2)
    return img2

for (lu, lv, ln, wd, rot, sd) in [(-0.16, -0.16, S*0.115, S*0.055, 0.7, 1),
                                  (0.40, 0.44, S*0.095, S*0.045, -0.5, 2),
                                  (-0.52, -0.10, S*0.080, S*0.038, 2.1, 3)]:
    img = leaf(S*0.5+lu*S*0.5, S*0.5+lv*S*0.5, ln, wd, rot, sd)

# ------------------------------------------------------------- seasoning
herb = smoothstep(0.30, 0.40, fbm(71, 16, 2))
herb = herb*(RAD < in_crust*0.96)
img += np.array([0.06, 0.09, 0.02], np.float32)[None, None, :]*herb[..., None]
flakes = smoothstep(0.34, 0.40, fbm(73, 10, 2))*inside
img += np.array([0.10, 0.09, 0.07], np.float32)[None, None, :]*flakes[..., None]

# ------------------------------------------------------------- steam
steam = np.zeros((S, S), np.float32)
for i in range(5):
    x0 = S*(0.34+0.09*i) + rng.normal(0, S*0.02)
    y0 = S*0.34
    for t in np.linspace(0, 1, 80):
        xx = x0 + np.sin(t*3.4+i)*S*0.035
        yy = y0 - t*S*0.22
        w = S*(0.03+0.05*t)
        steam += np.exp(-(((X-xx)/w)**2 + ((Y-yy)/(w*1.3))**2))*(1-t)*0.020
steam = gaussian_filter(steam, 10)
img += steam[..., None]*np.array([1.0, 0.95, 0.86], np.float32)

# ------------------------------------------------------------- grade
img = np.clip(img, 0, 4)
lum = img.mean(-1)
bloom = gaussian_filter(np.clip(lum-0.95, 0, None), 30)
img += bloom[..., None]*0.10*np.array([1.0, 0.88, 0.66], np.float32)
vig = 1-0.55*np.clip(RAD-0.62, 0, None)
img *= vig[..., None]
img *= 1.12
img = np.clip(img,0,None)**1.08
img = img/(1.0+0.10*img)
img[..., 0] *= 1.08
img[..., 2] *= 0.94
img = np.asarray(img, np.float32).reshape(S, S, 3)

out = Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8))
out = out.resize((1024, 1024), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2.0, 45, 3))
out.save('/workspace/taste-render/entry.png', optimize=True)
print('ok', out.size)
