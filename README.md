# RobustTrust-FL

**A Trust-Aware and Attack-Resilient Federated Learning Framework for
Scalable Privacy-Preserving Distributed Intelligence**

RobustTrust-FL is a modular federated learning framework designed for
heterogeneous, partially adversarial, and privacy-sensitive distributed
environments. It combines adversarial update detection, dynamic
client-trust estimation, straggler-aware coordination,
privacy-preserving model exchange, trust-aware robust aggregation, and
auditable decision logging in a unified federated learning pipeline.

The framework is intended for reproducible research on non-IID data
distributions, malicious clients, communication delays, privacy
constraints, and large-scale logical client populations in IoT, IIoT,
edge computing, and distributed cybersecurity settings.

## Key Components

-   **PrivacyGuard** --- applies update clipping, differential-privacy
    noise, privacy-budget tracking, and protected communication
    metadata.
-   **AttackShield** --- identifies suspicious updates using update
    deviation, cosine dissimilarity, gradient-norm anomalies, and
    temporal inconsistency.
-   **TrustShieldNet** --- estimates client trust from update quality,
    reliability, participation consistency, contribution effectiveness,
    adversarial safety, and historical trust.
-   **StragglerSync** --- accounts for delayed and stale client updates
    using freshness-aware coordination.
-   **Trust-Aware Robust Aggregator** --- combines client sample counts,
    trust, reliability, update freshness, update quality, and
    adversarial risk when weighting accepted updates.
-   **VerifiAudit** --- records round-level decisions and chained
    cryptographic hashes to support traceability and audit analysis.

## Framework Workflow

``` text
Global Model
    |
Trust-Aware Client Selection
    |
Local Client Training
    |
PrivacyGuard
    |
AttackShield
    |
TrustShieldNet
    |
StragglerSync
    |
Trust-Aware Robust Aggregation
    |
Candidate Model Validation / Rollback
    |
Updated Global Model
    |
VerifiAudit
```

Client updates may be evaluated under benign and malicious-client
scenarios. Candidate global models are checked before acceptance; if
validation fails, the previous verified global model is restored.

## Repository Structure

``` text
RobustTrust-FL/
├── README.md
├── requirements.txt
├── config/
│   ├── base.yaml
│   ├── demo.yaml
│   ├── main.yaml
│   ├── ciciot2023.yaml
│   ├── toniot.yaml
│   ├── edgeiiotset.yaml
│   ├── attacks.yaml
│   └── scalability.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── partitions/
├── robusttrust_fl/
│   ├── datasets/
│   ├── models/
│   ├── clients/
│   ├── privacy/
│   ├── attacks/
│   ├── security/
│   ├── trust/
│   ├── scheduling/
│   ├── aggregation/
│   ├── audit/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── results/
└── tests/
```

## Requirements

The recommended environment is:

-   Python 3.11
-   PyTorch 2.4 or a compatible version
-   NumPy
-   Pandas
-   Scikit-learn
-   SciPy
-   Matplotlib
-   PyYAML
-   Opacus
-   Cryptography
-   NetworkX
-   Flower
-   PyTest

Install the project dependencies from the repository root:

``` bash
python -m venv .venv
```

Activate the environment.

**Windows (Command Prompt):**

``` bat
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

``` bash
source .venv/bin/activate
```

Then install dependencies:

``` bash
pip install -r requirements.txt
```

## Quick Start

Run the synthetic demonstration configuration first. It is intended to
let you test the project without downloading an external dataset.

``` bash
python scripts/run_experiment.py --config config/demo.yaml
```

Run the main experiment configuration with:

``` bash
python scripts/run_experiment.py --config config/main.yaml
```

The commands above assume the corresponding scripts, configuration
files, and dependencies are present in your local checkout.

## Supported Datasets

### CICIoT2023

Used for adversarial and scalability-oriented federated learning
experiments.

Dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Expected location:

``` text
data/raw/ciciot2023/
```

### TON_IoT

Used for heterogeneous IoT/IIoT federated learning evaluation.

Dataset: https://research.unsw.edu.au/projects/toniot-datasets

Expected location:

``` text
data/raw/toniot/
```

### Edge-IIoTset

Used for industrial edge-intelligence, adversarial-client,
communication-delay, and straggler experiments.

Dataset:
https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot

Expected location:

``` text
data/raw/edgeiiotset/
```

Download datasets from their respective sources and review each
dataset's terms before use. Do not commit large datasets or data that
you are not authorized to redistribute.

## Data Preprocessing and Client Partitioning

The documented preprocessing workflow is:

1.  Remove duplicate records.
2.  Handle missing and infinite values.
3.  Encode categorical features.
4.  Process numeric features.
5.  Split data into training, validation, and test sets.
6.  Fit scaling transformations on training data only, then apply them
    to validation and test data.
7.  Partition training data among federated clients.

Both IID and non-IID client distributions are supported. The principal
non-IID approach uses Dirichlet partitioning.

Example configuration:

``` yaml
partition:
  type: dirichlet
  alpha: 0.5
  num_clients: 100
```

The documented guidance is that larger alpha values produce
distributions closer to IID, while smaller values produce stronger
client heterogeneity.

## Privacy, Security, and Trust

### PrivacyGuard

Documented configuration parameters include update-clipping threshold,
noise multiplier, maximum privacy budget, and delta. Privacy results
should be interpreted in the context of the selected configuration and
threat model.

### AttackShield

The documented risk indicators are: - Update deviation - Cosine
dissimilarity - Gradient-norm anomaly - Temporal inconsistency

The combined risk score is compared with a configured threshold to help
determine whether an update should be accepted.

### TrustShieldNet

Client trust is updated using current trust evidence and historical
trust. The framework defines trusted, uncertain, and restricted trust
ranges in its configuration/documentation.

### StragglerSync

Update freshness decreases with update age. Updates beyond the
configured maximum age are excluded. The simulator uses virtual arrival
times to represent delays rather than necessarily waiting for delayed
clients in real time.

### Trust-Aware Robust Aggregation

Accepted client updates are weighted using factors including sample
count, trust, reliability, freshness, update quality, and adversarial
risk. A maximum client-weight setting can limit an individual client's
influence.

### VerifiAudit

Round-level audit metadata may include selected, accepted, and rejected
clients; risk and trust scores; freshness scores; aggregation weights;
privacy status; candidate-model status; model hashes; and chained
digests. The design records compact decision metadata rather than raw
client datasets or unprotected full client updates.

## Supported Attack Scenarios

The project documentation lists the following attack scenarios:

-   Label-flipping attack
-   Sign-flipping attack
-   Gaussian update poisoning
-   Model-replacement attack
-   Byzantine attack
-   Backdoor attack

The documented attack-ratio experiments include malicious-client ratios
of 10%, 20%, 30%, and 40%.

Run the attack sweep with:

``` bash
python scripts/run_attack_sweep.py --config config/main.yaml
```

## Baselines and Experiments

Documented comparison methods include:

-   FedAvg
-   FedProx
-   SCAFFOLD
-   Coordinate-Wise Median
-   Trimmed Mean
-   Krum and Multi-Krum
-   Bulyan
-   Federated Differential Privacy
-   FLTrust
-   Reputation-Based FL
-   Trust-Based FL
-   RobustTrust-FL

Run the relevant experiment scripts when they are included in the
checkout:

``` bash
python scripts/compare_methods.py --config config/main.yaml
python scripts/run_noniid_sweep.py --config config/main.yaml
python scripts/run_privacy_sweep.py --config config/main.yaml
python scripts/run_ablation.py --config config/main.yaml
python scripts/run_scalability.py --config config/scalability.yaml
python scripts/run_repeated.py --config config/main.yaml
```

The documented ablation studies include removing AttackShield,
TrustShieldNet, StragglerSync, PrivacyGuard, VerifiAudit, or trust-aware
weighting.

For fair comparisons, keep dataset partitions, random seeds, local model
architecture, optimization settings, client participation, attack
assignments, communication conditions, and evaluation partitions
consistent unless a method requires a documented difference.

## Evaluation

The documented evaluation metrics include:

-   **Predictive performance:** accuracy, precision, recall, F1-score,
    ROC-AUC, and PR-AUC.
-   **Attack detection:** attack detection rate, false-positive rate,
    false-negative rate, malicious-detection precision and F1.
-   **Trust estimation:** trust classification accuracy, trust F1,
    ROC-AUC, score stability, Brier score, and expected calibration
    error.
-   **Synchronization:** round-completion latency, mean update age,
    stale-update ratio, and convergence rounds.
-   **Privacy:** privacy expenditure, epsilon, delta, and changes in
    predictive performance.
-   **Auditability:** verification success, trace completeness, tamper
    detection, storage per round, and audit overhead.
-   **Scalability and runtime:** aggregation time, total runtime,
    communication volume, peak memory, and round latency.

The documentation describes repeated runs with multiple random seeds and
summaries using means, standard deviations, confidence intervals, and
statistical tests.

## Tests

Run the test suite with:

``` bash
pytest -q
```

A focused AttackShield test can be run with:

``` bash
pytest tests/test_attackshield.py -q
```

These commands require the corresponding tests to be present in the
repository.

## Expected Outputs

Depending on the experiments executed, the documented outputs include:

-   Global accuracy and loss curves
-   Trust-score and adversarial-risk evolution
-   Attack-detection results
-   Malicious-client-ratio and non-IID sensitivity
-   Privacy--utility curves
-   Convergence and scalability analysis
-   Aggregation-weight analysis
-   Ablation results
-   Communication-cost and audit-overhead results
-   Statistical comparison results

Results are intended to be organized under `results/`, including raw
outputs, summaries, tables, figures, and audit records.

## Important Limitations

Federated learning alone does not guarantee complete privacy. Privacy
and security depend on the selected threat model, configuration, and
deployment environment. This framework should not be interpreted as
providing absolute protection against compromised endpoints, malicious
local software, side-channel attacks, unrestricted client-server
collusion, every model-inversion or membership-inference attack, or
attacks against released global models.

There is also a design trade-off between server-side client-update
analysis and secure aggregation: secure aggregation can hide individual
updates from the server, while server-side anomaly detection may require
client-level update statistics. The appropriate design depends on the
confidentiality requirements and threat model.

Scalability experiments use simulated logical clients; they do not by
themselves demonstrate deployment on the same number of physical
IoT/IIoT devices. Results depend on preprocessing, data partitions,
random seeds, hardware, attack settings, privacy parameters, model
architecture, and hyperparameters.

## License

No license is specified here. Select and add a `LICENSE` file before
public release once the intended permissions and redistribution
requirements have been decided. The project documentation mentions MIT
License and Apache License 2.0 as options to consider; make sure you
have the rights to license all included code and materials.

## Citation

If you use RobustTrust-FL in academic work, cite the associated paper
after its bibliographic details have been finalized.

The project documentation currently provides this placeholder:

``` bibtex
@article{robusttrustfl,
  title   = {RobustTrust-FL: A Trust-Aware and Attack-Resilient Federated Learning Framework for Scalable Privacy-Preserving Distributed Intelligence},
  author  = {Authors to be updated},
  journal = {Journal information to be updated},
  year    = {2026}
}
```

Replace the placeholder author and journal information with the final
publication details when available.

## Research Use

RobustTrust-FL is intended to support research in:

-   Federated learning
-   Trustworthy and privacy-aware machine learning
-   Adversarial federated learning
-   Distributed cybersecurity
-   IoT and IIoT security
-   Edge intelligence
-   Reproducibility and robustness benchmarking

## Contact

For questions, reproducibility issues, or academic collaboration, open
an issue in the GitHub repository.
