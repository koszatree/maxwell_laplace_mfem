import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, diags, bmat
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
import os

os.makedirs('output', exist_ok=True)


# 1. Parametry geometryczne

R = 25.0
V_IN = 0.0
V_OUT = 1.0
r1 = 1.4
r2 = 2.0
V_R = 0.0
GAP_DEG = 0.0


# 2. Siatka

def build_radii():
    r_in = np.linspace(0.0, r1, 8)
    r_gap = np.linspace(r1, r2, 7)[1:]
    K = int(np.ceil(np.log(R / r2) / np.log(1.08)))
    r_out = r2 * (R / r2) ** (np.arange(1, K + 1) / K)
    return np.concatenate([r_in, r_gap, r_out])

radii = build_radii()
dr = np.diff(radii)
pts_list = [np.array([[0.0, 0.0]])]
for k in range(1, len(radii)):
    rk = radii[k]
    h = dr[k - 1] if k == len(radii) - 1 else min(dr[k - 1], dr[k])
    nk = max(8, int(np.ceil(2 * np.pi * rk / (0.8 * h))))
    th = 2 * np.pi * np.arange(nk) / nk + 0.618 * k
    pts_list.append(np.c_[rk * np.cos(th), rk * np.sin(th)])
pts = np.vstack(pts_list)

tris = Delaunay(pts).simplices
_x0, _y0 = pts[tris[:, 0], 0], pts[tris[:, 0], 1]
_x1, _y1 = pts[tris[:, 1], 0], pts[tris[:, 1], 1]
_x2, _y2 = pts[tris[:, 2], 0], pts[tris[:, 2], 1]
_areas = np.abs(0.5 * ((_x1 - _x0) * (_y2 - _y0) - (_x2 - _x0) * (_y1 - _y0)))
tris = tris[_areas > 1e-12]

N_nodes = len(pts)
N_elements = len(tris)

edge_map = {}
edges = []
for tri_nodes in tris:
    for k in range(3):
        i0, i1 = tri_nodes[k], tri_nodes[(k + 1) % 3]
        edge = (i0, i1) if i0 < i1 else (i1, i0)
        if edge not in edge_map:
            edge_map[edge] = len(edges)
            edges.append(edge)
N_edges = len(edges)
print(f'Siatka: {N_nodes} węzłów, {N_elements} trójkątów, {N_edges} krawędzi')


# 3. Macierze elementowe

def laplace_matrix(nodes, tris):
    rr, cc, dd = [], [], []
    for ids in tris:
        x0, y0 = nodes[ids[0]]; x1, y1 = nodes[ids[1]]; x2, y2 = nodes[ids[2]]
        area = abs(0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)))
        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])
        ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for i in range(3):
            for j in range(3):
                rr.append(ids[i]); cc.append(ids[j]); dd.append(ke[i, j])
    return csr_matrix((dd, (rr, cc)), shape=(len(nodes), len(nodes)))

def build_grad_matrix_edges_nodes(pts, edge_map):
    M = len(edge_map); N = len(pts)
    row, col, val = [], [], []
    for (i_a, i_b), e_idx in edge_map.items():
        L = np.linalg.norm(pts[i_b] - pts[i_a])
        row += [e_idx, e_idx]; col += [i_a, i_b]; val += [-1.0 / L, 1.0 / L]
    return csr_matrix((val, (row, col)), shape=(M, N))

def build_mass_matrix_edges(pts, tris, edge_map):
    M = len(edge_map)
    w = np.zeros(M)
    for ids in tris:
        for k in range(3):
            o = ids[k]; a = ids[(k + 1) % 3]; b = ids[(k + 2) % 3]
            u = pts[a] - pts[o]; v = pts[b] - pts[o]
            cot = np.dot(u, v) / abs(u[0] * v[1] - u[1] * v[0])
            e = pts[b] - pts[a]
            edge = (a, b) if a < b else (b, a)
            w[edge_map[edge]] += 0.5 * cot * np.dot(e, e)
    return diags(w, offsets=0, shape=(M, M))

G = build_grad_matrix_edges_nodes(pts, edge_map)
M_edge = build_mass_matrix_edges(pts, tris, edge_map)
K_nt = laplace_matrix(pts, tris)
print('||K_nt - G^T M_edge G||_max =', abs(K_nt - G.T @ M_edge @ G).max())


# 4. Warunki brzegowe

r_node = np.hypot(pts[:, 0], pts[:, 1])
tol = 1e-9
inner = r_node <= r1 + tol
theta_node = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
outer = (np.abs(r_node - r2) < tol) & (np.abs(theta_node) >= GAP_DEG / 2.0)  # ZMIANA 8
bound = r_node >= R - tol

V_fixed = np.zeros(N_nodes)
V_fixed[inner] = V_IN
V_fixed[outer] = V_OUT
V_fixed[bound] = V_R
fixed_mask = inner | outer | bound
fixed_nodes = np.where(fixed_mask)[0]
free_nodes = np.where(~fixed_mask)[0]


# 5. Układ mieszany

Gf = G[:, free_nodes].tocsr()
Gx = G[:, fixed_nodes].tocsr()
A_mixed = bmat([
    [M_edge,        M_edge @ Gf],
    [Gf.T @ M_edge, None],
], format='csc')
rhs = np.concatenate([-(M_edge @ (Gx @ V_fixed[fixed_nodes])),
                      np.zeros(len(free_nodes))])
sol = spsolve(A_mixed, rhs)
E_edge = sol[:N_edges]
V_node = np.zeros(N_nodes)
V_node[free_nodes] = sol[N_edges:]
V_node[fixed_nodes] = V_fixed[fixed_nodes]
print('Kontrola E = -G V:', np.abs(E_edge + G @ V_node).max())


# 6. Rekonstrukcja E w trójkątach

eps0 = 1.0
E_tri_x = np.zeros(N_elements); E_tri_y = np.zeros(N_elements)
for tidx, ids in enumerate(tris):
    dirs, vals = [], []
    for k in range(3):
        i_a, i_b = ids[k], ids[(k + 1) % 3]
        edge = (i_a, i_b) if i_a < i_b else (i_b, i_a)
        vec = pts[edge[1]] - pts[edge[0]]
        dirs.append(vec / np.linalg.norm(vec)); vals.append(E_edge[edge_map[edge]])
    E_xy, *_ = np.linalg.lstsq(np.array(dirs), np.array(vals), rcond=None)
    E_tri_x[tidx], E_tri_y[tidx] = E_xy
E_mag_tri = np.hypot(E_tri_x, E_tri_y)
D_elem = eps0 * E_mag_tri


# 7. Gęstość energii

triang = Triangulation(pts[:, 0], pts[:, 1], tris)
finder = triang.get_trifinder()
eps0_vacc = 8.8541878128e-12

def u_on_grid(xg, yg):
    idx = finder(xg, yg)
    Em = np.where(idx >= 0, E_mag_tri[np.maximum(idx, 0)], np.nan)
    return 0.5 * eps0_vacc * Em**2

xx_out, yy_out = np.meshgrid(np.linspace(-R, R, 300), np.linspace(-R, R, 300))
u_grid = u_on_grid(xx_out, yy_out)
u_grid = np.where(xx_out**2 + yy_out**2 <= R**2, u_grid, np.nan)

fig, ax = plt.subplots(1, 1, figsize=(7, 6), dpi=130)
with np.errstate(divide='ignore'):
    lg = np.log10(np.where(u_grid > 0, u_grid, np.nan))
cf = ax.contourf(xx_out, yy_out, lg, levels=50, cmap='magma')
ax.set_aspect('equal'); fig.colorbar(cf, ax=ax, label='log10 u [J/m³]')
ax.set_title('Gęstość energii elektrostatycznej')
plt.tight_layout(); plt.savefig('output/MFEM_Maxwell_gestosc.png'); plt.close()

N_compare = 50
xx_cmp, yy_cmp = np.meshgrid(np.linspace(-R, R, N_compare), np.linspace(-R, R, N_compare))
u_cmp = np.nan_to_num(u_on_grid(xx_cmp, yy_cmp), nan=0.0)
u_cmp = np.where(xx_cmp**2 + yy_cmp**2 <= R**2, u_cmp, 0.0)
pd.DataFrame(u_cmp).to_csv('output/u_mixed_Maxwell.csv', header=False, index=False)


def V_analytic(r, r1, r2, R, V0=1.0):
    r = np.asarray(r, dtype=np.float64)
    V = np.zeros_like(r)
    m1 = (r >= r1) & (r <= r2)
    m2 = (r > r2) & (r <= R)
    V[m1] = V0 * np.log(r[m1] / r1) / np.log(r2 / r1)
    V[m2] = V0 * np.log(R / r[m2]) / np.log(R / r2)
    return V

m_L2 = (r_node >= r1 - tol) & (r_node <= R + tol)
V_ana_nodes = V_analytic(r_node, r1, r2, R, V_OUT - V_IN)
err_V_mfem = np.sqrt(np.mean((V_node[m_L2] - V_ana_nodes[m_L2])**2)
                     / np.mean(V_ana_nodes[m_L2]**2))
print(f"\nBłąd względny L2 potencjału (węzły siatki, r1<r<R): {err_V_mfem:.3e}")


# 8. Pojemność kondensatora

dV_cap = V_OUT - V_IN

tx0, ty0 = pts[tris[:, 0], 0], pts[tris[:, 0], 1]
tx1, ty1 = pts[tris[:, 1], 0], pts[tris[:, 1], 1]
tx2, ty2 = pts[tris[:, 2], 0], pts[tris[:, 2], 1]
area_tri = np.abs(0.5 * ((tx1 - tx0) * (ty2 - ty0) - (tx2 - tx0) * (ty1 - ty0)))

r_tri_nodes = r_node[tris]
in_gap = np.all((r_tri_nodes >= r1 - tol) & (r_tri_nodes <= r2 + tol), axis=1)

integral_E2 = np.sum(area_tri[in_gap] * E_mag_tri[in_gap]**2)
W_energy = 0.5 * eps0_vacc * integral_E2
C_energy = 2.0 * W_energy / dV_cap**2

def capacitance_flux_mfem(r0, n_th=2000):
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    xs, ys = r0 * np.cos(th), r0 * np.sin(th)
    idx = finder(xs, ys)
    ok = idx >= 0
    Er = E_tri_x[idx[ok]] * np.cos(th[ok]) + E_tri_y[idx[ok]] * np.sin(th[ok])
    Q = eps0_vacc * np.mean(Er) * 2 * np.pi * r0
    return abs(Q) / dV_cap

r_mid = np.sqrt(r1 * r2)
C_flux = capacitance_flux_mfem(r_mid)

nodal_flux = G.T @ (M_edge @ E_edge)
Q_inner = eps0_vacc * abs(np.sum(nodal_flux[inner]))
C_charge = Q_inner / dV_cap

C_ana = 2 * np.pi * eps0_vacc / np.log(r2 / r1)

print("\nPojemność na jednostkę długości (MFEM Maxwell):")
print(f"  analityczna:                          {C_ana:.6e} F/m")
print(f"  energia (suma po trójkątach):         {C_energy:.6e} F/m  (błąd wzgl. {abs(C_energy - C_ana) / C_ana:.2%})")
print(f"  strumień (okrąg r0={r_mid:.3f}):          {C_flux:.6e} F/m  (błąd wzgl. {abs(C_flux - C_ana) / C_ana:.2%})")
print(f"  ładunek elektrody (reakcje węzłowe):  {C_charge:.6e} F/m  (błąd wzgl. {abs(C_charge - C_ana) / C_ana:.2%})")

print("\nStałość strumienia w szczelinie:")
for r0 in np.linspace(r1 * 1.05, r2 * 0.95, 5):
    print(f"  r0={r0:.3f}: C_flux={capacitance_flux_mfem(r0):.6e} F/m")