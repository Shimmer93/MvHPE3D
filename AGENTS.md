# AGENTS.md

## 1. Project Description

This repository is for a research project on calibration-free multi-view SMPL fusion from single-view SAM3DBody predictions.

Preferred training framework:
- PyTorch Lightning

Current Stage 1 scope:
- Input per view: `mhr_model_params + shape_params`
- Fusion: simple MLP-based, permutation-invariant fusion
- Target: fused canonical body in a pelvis-centered, SMPL root rotation-removed space

Current Stage 2 scope:
- Precompute offline per-view fitted SMPL from MHR compact parameters
- Input per view: canonical fitted SMPL `body_pose + betas`
- Internal pose representation: per-joint 6D rotations for fusion/refinement
- Initialization: weighted permutation-invariant fusion in canonical parameter space
- Prediction: iterative residual refinement of canonical SMPL parameters
- Target: fused canonical body in the same pelvis-centered, SMPL root rotation-removed space as Stage 1

Out of scope for Stage 1:
- Learned per-view camera prediction
- Strong geometry claims beyond canonical-body fusion

Auxiliary fields such as `smpl_global_orient`, `pred_cam_t`, and `cam_int` may still be used for qualitative visualization or camera-frame evaluation, but they are not primary Stage 2 training inputs.

## 2. Python Environment

Current repo signals:
- Python requirement: `>=3.12,<3.13`
- Workspace tooling: `uv` with checked-in `uv.lock`
- Preferred training stack: PyTorch Lightning
- The modified SAM3DBody code is tracked as a git submodule at `external/sam-3d-body`

Current dependency baseline:
- `torch`
- `pytorch-lightning`
- `numpy`
- `opencv-python`
- `pyrootutils`
- `detectron2`
- `moge`
- `mhr`
- `pymomentum-gpu`

Recommended environment workflow:
- initialize submodules: `git submodule update --init --recursive`
- sync dependencies: `uv sync`
- run one-off commands: `uv run <command>`
- enter the environment: `source .venv/bin/activate`
- verify Python version: `uv run python --version`
- run repo tests: `uv run pytest -q tests`

Environment note:
- the checked-in environment is defined by `pyproject.toml` plus `uv.lock`
- `sam3` is not currently declared in the main project dependencies; install it separately only if you need the optional SAM3 detector/segmentor paths in `external/sam-3d-body`

Current pinned interpreter range:
- Python `3.12.x`

Update this section further once training, evaluation, and visualization entrypoints are implemented.

## 3. File Structure

Planned structure:

```text
MvHPE3D/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── external/
│   └── sam-3d-body/
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── humman_stage1.yaml
│   │   └── humman_stage2.yaml
│   ├── model/
│   │   ├── stage1_mlp.yaml
│   │   └── stage2_param_refine.yaml
│   ├── trainer/
│   │   └── default.yaml
│   └── experiment/
│       ├── stage1_cross_camera.yaml
│       └── stage2_cross_camera.yaml
├── scripts/
│   ├── precompute_input_smpl.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── visualize.py
├── src/
│   └── mvhpe3d/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datamodule.py
│       │   ├── datasets/
│       │   │   ├── humman_multiview.py
│       │   │   └── humman_stage2_multiview.py
│       │   ├── splits.py
│       │   ├── collate.py
│       │   └── canonicalization.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── components/
│       │   │   ├── mlp.py
│       │   │   └── deepsets.py
│       │   ├── stage1/
│       │   │   └── mlp_fusion.py
│       │   └── stage2/
│       │       └── param_refine.py
│       ├── lightning/
│       │   ├── __init__.py
│       │   ├── stage1_module.py
│       │   └── stage2_module.py
│       ├── losses/
│       │   └── smpl_loss.py
│       ├── metrics/
│       │   ├── mpjpe.py
│       │   ├── pa_mpjpe.py
│       │   └── mve.py
│       ├── visualization/
│       │   ├── canonical.py
│       │   └── overlay.py
│       └── utils/
│           ├── io.py
│           ├── logging.py
│           ├── rotation.py
│           └── seed.py
└── tests/
    ├── test_canonicalization.py
    ├── test_dataset.py
    └── test_stage1_model.py
```

File ownership guidance:
- `external/sam-3d-body/`: pinned external dependency containing the modified SAM3DBody exporter and inference code
- `configs/`: experiment configuration, split into data, model, trainer, and composed experiments
- `scripts/`: thin entrypoints only; `precompute_input_smpl.py` is the offline bridge from compact MHR outputs to cached per-view fitted SMPL for Stage 2
- `src/mvhpe3d/data/`: dataset definitions, split logic, collation, and canonicalization
- `src/mvhpe3d/models/`: pure PyTorch model components and stage-specific architectures; Stage 2 should stay in parameter space unless there is a strong reason to move to mesh-state learning
- `src/mvhpe3d/lightning/`: LightningModule wrappers for training, validation, testing, logging, and optimizer setup
- `src/mvhpe3d/visualization/`: canonical renders and image overlays; overlay logic stays separate from learned model code
- `tests/`: unit tests for canonicalization, dataset schema, and model behavior

This structure is intentionally a bit ahead of the current scaffold so the codebase has room to grow without mixing data logic, model math, and Lightning orchestration.

## 4. Documentation

Primary documentation:
- `README.md` is the source of truth for the current project outline

Documentation expectations:
- Keep the README aligned with the actual implemented scope
- Document experimental assumptions explicitly, especially canonicalization, inputs, targets, and evaluation
- When adding scripts or modules, prefer brief docstrings and clear CLI help over long inline comments
- If a visualization uses a workaround or proxy signal, state that clearly in the docs
- Keep config names descriptive and make experiment entrypoints map cleanly to config files

## 5. Git

Working guidance:
- Keep changes small and scoped
- Avoid destructive history edits unless explicitly requested
- Do not amend commits unless explicitly requested
- Do not revert unrelated user changes
- Prefer clear commit messages that describe the actual change

Since this is a research repo, it is acceptable for documentation and code structure to evolve quickly, but the current README and code should not contradict each other.

## 6. Engineering Principle

Default principles for this repository:
- Keep Stage 1 simple and defensible before adding more ambitious modeling
- Make Stage 2 a parameter-space refinement baseline before considering mesh-state methods
- Prefer a clear baseline over premature architectural complexity
- Reduce ambiguity in coordinate systems, targets, and evaluation before optimizing model design
- Separate learned outputs from visualization tricks in both code and docs
- Make claims match the implementation
- Preserve permutation invariance when multi-view input ordering should not matter
- Prefer offline cached expensive preprocessing over online fitting inside the training loop when it keeps experiments reproducible
- Keep pure model code separate from PyTorch Lightning training orchestration

When in doubt, choose the simplest design that makes the experiment easy to reproduce and easy to falsify.
