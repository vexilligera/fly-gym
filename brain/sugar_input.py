"""Released unilateral sugar-GRN assay; no ingestion or reward dynamics."""
import numpy as np

# Eon a3db62f, code/benchmark.py EXPERIMENTS['sugar'] (v783 IDs).
SUGAR_IDS = (
    '720575940624963786', '720575940630233916', '720575940637568838',
    '720575940638202345', '720575940617000768', '720575940630797113',
    '720575940632889389', '720575940621754367', '720575940621502051',
    '720575940640649691', '720575940639332736', '720575940616885538',
    '720575940639198653', '720575940639259967', '720575940617937543',
    '720575940632425919', '720575940633143833', '720575940612670570',
    '720575940628853239', '720575940629176663', '720575940611875570',
)
# Released example.ipynb MN9 IDs, with sides from current v783 annotations.
MN9_IDS = {'left': '720575940618238523', 'right': '720575940660219265'}


class SugarInput:
    def __init__(self, ids, annotations):
        index = {rid: i for i, rid in enumerate(ids)}
        self.receptors = np.array([index[rid] for rid in SUGAR_IDS], dtype=np.int32)
        self.motor = {side: index[rid] for side, rid in MN9_IDS.items()}
        for rid in SUGAR_IDS:
            a = annotations[rid]
            if a['cell_class'] != 'gustatory' or a['cell_sub_class'] != 'sugar/water':
                raise ValueError('Sugar GRN annotation mismatch')
        for side, rid in MN9_IDS.items():
            a = annotations[rid]
            if a['cell_type'] != 'CB0701' or a['side'] != side:
                raise ValueError('MN9 v783 annotation mismatch')
        self.metadata = {
            'input_ids': list(SUGAR_IDS), 'input_neurons': len(SUGAR_IDS),
            'input': 'Released 21-cell sugar-GRN cohort; v783 LB3, left, sugar/water annotations',
            'motor_ids': MN9_IDS, 'motor': 'MN9 / CB0701, readout only; not directly stimulated',
            'source': 'Eon a3db62f code/benchmark.py sugar experiment and paper-phil-drosophila/example.ipynb',
            'paper': 'https://doi.org/10.1038/s41586-024-07763-9',
            'encoding': '0–200 Hz imposed sensory input, not a sucrose concentration calibration',
            'limits': 'Feeding-initiation circuit assay; no proboscis mechanics, ingestion, satiety, reward learning, or happiness measure',
        }

    def readout(self, delta, rate):
        return {'input_rate_hz': float(rate),
                'GRN_hz': float(delta[self.receptors].mean() / .02),
                'MN9_hz': {side: float(delta[i] / .02) for side, i in self.motor.items()}}
