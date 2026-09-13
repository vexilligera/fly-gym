"""A normalized, steady 2-D diffusion/decay field with impermeable maze walls.

This is an airborne food-odor proxy co-located with sugar, not evaporating
sucrose, turbulent CFD, or molecular receptor chemistry. Only local sensor
samples are supplied to the controller.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class OdorField:
    def __init__(self, walls, extent=20.5, spacing=.5, diffusion=12.0, decay=.12):
        self.axis = np.arange(-extent, extent + spacing/2, spacing)
        self.spacing = spacing
        x, y = np.meshgrid(self.axis, self.axis)
        blocked = np.zeros(x.shape, dtype=bool)
        for cx, cy, hx, hy in walls:
            blocked |= (np.abs(x-cx) <= hx) & (np.abs(y-cy) <= hy)
        blocked |= (np.abs(x) >= 20) | (np.abs(y) >= 20)
        self.blocked = blocked
        free = ~blocked
        source = (x*x + y*y < 1.5**2) & free
        mapping = np.full(x.shape, -1, dtype=int)
        mapping[free] = np.arange(free.sum())
        rows, cols, values = [], [], []
        rhs = np.zeros(free.sum())
        coefficient = diffusion / spacing**2
        for iy, ix in zip(*np.nonzero(free)):
            i = mapping[iy, ix]
            if source[iy, ix]:
                rows.append(i); cols.append(i); values.append(1.0); rhs[i] = 1.0
                continue
            diagonal = decay
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny, nx = iy+dy, ix+dx
                if 0 <= ny < len(self.axis) and 0 <= nx < len(self.axis) and free[ny,nx]:
                    rows.append(i); cols.append(mapping[ny,nx]); values.append(-coefficient)
                    diagonal += coefficient
            # No neighbor through a wall: zero normal flux, not an odor sink.
            rows.append(i); cols.append(i); values.append(diagonal)
        matrix = coo_matrix((values, (rows, cols)), shape=(free.sum(),free.sum())).tocsr()
        self.concentration = np.zeros_like(x)
        self.concentration[free] = spsolve(matrix, rhs)
        if not np.isfinite(self.concentration).all() or self.concentration.min() < -1e-9:
            raise ValueError('Invalid diffusion solution')
        self.concentration = np.clip(self.concentration, 0, 1)
        self.metadata = {'spacing_mm': spacing, 'diffusion_mm2_s': diffusion, 'decay_per_s': decay,
                         'units': 'normalized food-odor proxy, not ppm',
                         'model': 'Pre-equilibrated 2-D diffusion with decay, fixed source concentration, no-flux walls; no airflow or vertical mixing.'}

    def sample(self, positions):
        result = []
        for px, py in np.asarray(positions)[:, :2]:
            gx, gy = (np.array([px,py]) - self.axis[0]) / self.spacing
            ix, iy = int(np.floor(gx)), int(np.floor(gy))
            sx, sy = gx-ix, gy-iy
            total, weight = 0.0, 0.0
            for dx, dy, w in ((0,0,(1-sx)*(1-sy)),(1,0,sx*(1-sy)),(0,1,(1-sx)*sy),(1,1,sx*sy)):
                xx, yy = ix+dx, iy+dy
                if 0 <= xx < len(self.axis) and 0 <= yy < len(self.axis) and not self.blocked[yy,xx]:
                    total += w*self.concentration[yy,xx]; weight += w
            result.append(total / weight if weight > 1e-10 else 0.0)
        return np.array(result)

    def display(self):
        return {'axis_mm': self.axis.tolist(), 'values': self.concentration.round(4).tolist(),
                'blocked': self.blocked.astype(int).tolist(), 'metadata': self.metadata}
