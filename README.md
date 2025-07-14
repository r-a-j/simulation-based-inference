# Simulation-based inference for a leaky integrate-and-fire (LIF) neuron

<div align="center">
<img src="https://github.com/r-a-j/simulation-based-inference/blob/main/neuron.png" width="300">
</div>

*BayesFlow* + TensorFlow implementation of a likelihood-free inference pipeline
for a single-compartment **leaky integrate-and-fire (LIF) neuron**.

* **Forward model** Uniform priors over six biophysical parameters  
  \(C_m, g_L, E_L, V_\text{th}, V_\text{res}, t_\text{ref}\)  
  + coloured-noise current stimulus.
* **Inference** CouplingFlow posterior with TimeSeries summary network
  (1-channel voltage trace, 501 time steps, 10 000 simulated traces).
* **Diagnostics** Posterior-mean recovery, SBC ECDF, contraction/z-score.
* **Outputs** All artefacts saved under `results/<run_id>/…`.

---

## 1 · Quick start (Recommended Linux)

```bash
# clone and install
git clone https://github.com/r-a-j/simulation-based-inference.git
cd simulation-based-inference
python -m venv
pip install -r requirements.txt
```

- run the full offline workflow in jupyter notebook `main.ipynb` (≈ 12 min on a mid-range GPU Eg. Nvidia RTX 3070 8 GB Laptop GPU)

## 2 · Method details

| Symbol         | Prior range  | Meaning                          |
| -------------- | ------------ | -------------------------------- |
| $C_m$          | 100 – 200 pF | Membrane capacitance             |
| $g_L$          | 5 – 15 nS    | Leak conductance                 |
| $E_L$          | –80 … –65 mV | Resting potential                |
| $V_\text{th}$  | –60 … –50 mV | Spike threshold                  |
| $V_\text{res}$ | –80 … –70 mV | Reset potential (forced < $E_L$) |
| $t_\text{ref}$ | 1 … 5 ms     | Refractory time                  |

*Stimulus* white-noise current low-pass filtered with $τ=10\;ms$.

*Adapter* Standardises parameters & observables, concatenates into
`inference_variables`.

*Neural nets* Configuration / Coupling Flow with default BayesFlow settings;
TimeSeriesNetwork \[Conv + GRU].

## 3 · Results

| Metric                                                   | Value (500 posterior draws on 400 fresh traces)                                                |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Pearson-r $E_L, V_\text{th}, V_\text{res}, t_\text{ref}$ | ≥ 0.99                                                                                         |
| Pearson-r $C_m$                                          | 0.43                                                                                           |
| Pearson-r $g_L$                                          | 0.05                                                                                           |
| SBC                                                      | All four voltage parameters within 95 % bands;<br>passive parameters show mild mis-calibration |
| Posterior-predictive check                               | Observed trace falls inside 90 % predictive band                                               |

---
