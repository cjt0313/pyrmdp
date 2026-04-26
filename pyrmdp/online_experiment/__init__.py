"""
Online Update Experiment
========================

A client/server testbed for the Neuro-Symbolic online-update pipeline:

- Backend (PC):  FastAPI + pyrmdp + pyPPDDL.  Holds the abstract transition
  graph, Dirichlet posterior counts, Tabu ledger, and triggers MSCA
  re-synthesis when the Wasserstein spectral distance spikes.
- Frontend (laptop):  Streamlit.  Teleoperator view (Human 1) reports
  execution outcomes, expert diagnostics view categorises failures, the
  baseline view logs Human 2's manually-guessed recovery skills, and a
  dashboard plots live graph statistics.

The modules are intentionally skeletal — they expose the REST contract and
wire the mathematical primitives already implemented in
:mod:`pyrmdp.synthesis.iterative_synthesizer` so the experiment logic can
be iterated on without redoing plumbing.
"""
