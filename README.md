# RSU Deployment with K-center Approximation

This repository is a cleaned-up version of a course project for `Advanced Algorithms for Communications and Information Security`.

It explores a practical question: how to place a limited number of Roadside Units (RSUs) so that service coverage is as balanced as possible. The main algorithmic focus is the K-center problem, using Gonzalez's greedy 2-approximation as the core method.

## What is included

- A synthetic Euclidean-space experiment comparing:
  - Gonzalez greedy K-center
  - A random baseline
- A road-network experiment using OSMnx / OpenStreetMap data comparing:
  - Gonzalez greedy K-center
  - Greedy K-median
  - A random baseline
- Optional CSV result export for local inspection

## What is not included

- Course report PDF
- Presentation PDF
- Original Google Colab notebook

Those files are intentionally excluded to keep the public repository focused, smaller, and lower risk from a privacy / disclosure perspective.

## Repository files

- `rsu_deployment_k_center.py`: main script
- `requirements.txt`: Python dependencies
- `.gitignore`: excludes local caches, generated outputs, notebooks, and course artifacts

## How to run

Use Python 3.10+ in a clean virtual environment.

```bash
pip install -r requirements.txt
python rsu_deployment_k_center.py --skip-network --no-plots
```

The command above runs only the synthetic experiment.

To run the road-network experiment as well:

```bash
python rsu_deployment_k_center.py --network-k-values 5 10 15 --max-nodes 120
```

To save CSV summaries and generated plots locally:

```bash
python rsu_deployment_k_center.py --save-results
```

Generated files will go into `outputs/`, which is ignored by Git.

## Notes on the road-network experiment

- The script queries road-network data from OpenStreetMap through OSMnx.
- Results depend on the map snapshot, chosen center point, query radius, and sampled node count.
- To keep all-pairs shortest-path computation manageable, the script samples at most `--max-nodes` connected road nodes before building the distance matrix.

## Scope and limitations

- This is a course-project algorithm experiment, not a production deployment system.
- The implementation is meant to demonstrate modeling, approximation algorithms, and experiment setup.
- It does not include operational constraints such as RSU hardware limits, traffic demand modeling, radio propagation, or deployment cost optimization beyond the simple choice of `k`.

## Privacy and security posture

- No report PDF, presentation PDF, or notebook metadata is included.
- No personal student ID, local file paths, private cloud links, credentials, or tokens are intended to be present in this repository.
- OpenStreetMap data is fetched at runtime; no bundled map dump or sensitive local dataset is included.

## Suggested résumé / portfolio framing

If you reference this project in a résumé or interview, a plain description like this is accurate:

`Implemented and evaluated K-center-based heuristics for RSU placement, including synthetic and road-network experiments using Python, NetworkX, and OSMnx.`
