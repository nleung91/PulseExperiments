from slab.experiments.PulseExperiments.sequences import PulseSequences
from slab.experiments.PulseExperiments.pulse_experiment import Experiment
import numpy as np

def histogram(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['histogram']
    for amp in np.arange(expt_cfg['amp_start'], expt_cfg['amp_stop'], expt_cfg['amp_step']):

        experiment_cfg['histogram']['amp'] = amp

        ps = PulseSequences(quantum_device_cfg , experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')

        exp = Experiment(quantum_device_cfg , experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'histogram')
