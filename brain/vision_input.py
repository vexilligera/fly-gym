"""Explicitly artificial retinotopy for a closed-loop visual experiment.

The v783 annotations bundled here contain no visual-field/column registration.
We assign L2 cells to 721 pixels per eye deterministically, and assign R1–6 to
the pixel of their strongest released L2 connection. This is a synthetic
registration, NOT retinotopy inferred from annotation anchor coordinates.
All released signed connections remain unchanged in the recurrent simulator.
"""
import numpy as np


class VisualInput:
    def __init__(self, ids, annotations, pre, post, weights):
        types = np.array([annotations.get(rid, {}).get('cell_type', '') for rid in ids])
        sides = np.array([annotations.get(rid, {}).get('side', '') for rid in ids])
        self.receptors = np.flatnonzero((types == 'R1-6') & np.isin(sides, ['left', 'right']))
        self.readout = np.flatnonzero((types == 'L2') & np.isin(sides, ['left', 'right']))
        pixel = np.full(len(ids), -1, dtype=np.int32)
        rng = np.random.default_rng(783)
        for eye, side in enumerate(['left', 'right']):
            cells = self.readout[sides[self.readout] == side]
            pixel[rng.permutation(cells)] = eye * 721 + np.arange(len(cells)) % 721
        receptor_mask = types == 'R1-6'
        side_code = np.where(sides == 'left', 0, np.where(sides == 'right', 1, 2)).astype(np.uint8)
        valid = receptor_mask[pre] & (pixel[post] >= 0) & (side_code[pre] == side_code[post])
        strongest = np.zeros(len(ids))
        for a, b, weight in zip(pre[valid], post[valid], weights[valid]):
            if abs(weight) > strongest[a]:
                strongest[a] = abs(weight)
                pixel[a] = pixel[b]
        linked = int((pixel[self.receptors] >= 0).sum())
        for eye, side in enumerate(['left', 'right']):
            unlinked = self.receptors[(sides[self.receptors] == side) & (pixel[self.receptors] < 0)]
            pixel[unlinked] = eye * 721 + np.arange(len(unlinked)) % 721
        self.receptor_pixels = pixel[self.receptors]
        self.readout_pixels = pixel[self.readout]
        # The control disrupts spatial assignment, including left/right, while
        # retaining every retinal sample and the same mapping for decoding.
        self.shuffled_pixels = np.random.default_rng(2026).permutation(1442)
        self.metadata = {
            'schema': 'synthetic-r1-l2-retinotopy-v1',
            'ommatidia_per_eye': 721, 'input_neurons': len(self.receptors),
            'readout_neurons': len(self.readout), 'receptors_linked_to_L2': linked,
            'mapping_seed': 783, 'shuffle_seed': 2026,
            'registration': 'Artificial L2-to-pixel assignment; R1–6 follow their strongest released L2 connection. No measured retinal registration.',
            'encoding': 'Dark contrast → 0–180 Hz Poisson input to R1–6 LIF cells. This bypasses graded phototransduction and histamine chemistry.',
            'decoder': 'Actual postsynaptic L2 spike counts → artificial population-vector bearing → differential CPG gains. No natural descending visual policy is claimed.',
            'sign_caveat': 'Released signs are retained, including many excitatory R1–6 outputs; real photoreceptors signal with graded histamine release. This is not a validated photoreceptor/lamina model.',
        }

    def rates(self, contrast, condition='vision'):
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != (2, 721) or not np.isfinite(contrast).all():
            raise ValueError('Expected finite contrast with shape (2, 721)')
        if condition not in ('vision', 'blind', 'shuffled', 'readout_off'):
            raise ValueError('Unknown visual control condition')
        pixels = contrast.ravel()
        if condition == 'blind':
            pixels = np.zeros_like(pixels)
        elif condition == 'shuffled':
            pixels = pixels[self.shuffled_pixels]
        return 180 * np.clip(pixels[self.receptor_pixels], 0, 1)
