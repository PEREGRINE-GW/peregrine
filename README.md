<img align="center" height="200" src="./images/peregrine_logo.png">

[![version](https://img.shields.io/badge/version-0.0.1-blue)](https://github.com/PEREGRINE-GW/peregrine) [![DOI](https://img.shields.io/badge/DOI-arXiv.2304.02035-brightgreen)](https://arxiv.org/abs/2304.02035)

## v0.0.2 available soon | currently under development (use cbc or overlapping branch for now)

## Description

- **PEREGRINE** is a Simulation-based Inference (SBI) library designed to perform analysis on a wide class of gravitational wave signals. It is built on top of the [swyft](https://swyft.readthedocs.io/en/) code, which implements neural ratio estimation to efficiently access marginal posteriors for all parameters of interest.
- **Related paper:** The details regarding the implementation of the TMNRE algorithm and the specific demonstration for compact binary black hole mergers can be found in [arxiv:2304.02035](https://arxiv.org/abs/2304.02035).
- **Key benefits:** We showed in the above paper that PEREGRINE is extremely sample efficient compared to traditional methods - e.g. for a BBH merger, we required only 2% of the waveform evaluations than common samplers such as dynesty. The method is also an 'implicit likelihood' technique, so it inherits all the associated advantages such as the fact that it does not require an explicit likelihood to be written down. This opens up the possibility of using PEREGRINE to analyse a wide range of transient or continuous gravitational wave sources.
- **Contacts:** For questions and comments on the code, please contact either [Uddipta Bhardwaj](mailto:u.bhardwaj@uva.nl) or [James Alvey](mailto:j.b.g.alvey@uva.nl). Alternatively feel free to open an issue.
- **Citation:** If you use PEREGRINE in your analysis, or find it useful, we would ask that you please use the following citation.
```
@article{Bhardwaj:2023xph,
    author = "Bhardwaj, Uddipta and Alvey, James and Miller, Benjamin Kurt and Nissanke, Samaya and Weniger, Christoph",
    title = "{Peregrine: Sequential simulation-based inference for gravitational wave signals}",
    eprint = "2304.02035",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "4",
    year = "2023"
}
```

## Available Branches:
- `cbc` - analysis for 2G detector single GWs is implemented
- `overlapping` - analysis for multiple GWs in a 2G detector

## Release Details:
- v0.0.1 | *August 2023* | Public PEREGRINE release matching companion paper:
    - [Peregrine: Sequential simulation based inference for gravitational waves](https://arxiv.org/abs/2304.02035)
    - [What to do when things get crowded? Scalable joint analysis of overlapping gravitational wave signals](https://arxiv.org/abs/2308.06318)
- v0.0.2 | *March 2024* | Coming soon!
