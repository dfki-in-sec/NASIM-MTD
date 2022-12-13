# NASIM-MTD

This repository contains the official code for the following paper:

**Daniel Reti, Daniel Fraunholz, Karina Elzer, Daniel Schneider, and Hans Dieter Schotten. 2022. Evaluating Deception and Moving Target Defense with Network Attack Simulation. In Proceedings of the 9th ACM Workshop on Moving Target Defense (MTD'22). Association for Computing Machinery, New York, NY, USA, 45–53. https://doi.org/10.1145/3560828.3564006**

See the [README](README.rst) of the code for more information.

## Credits
The code is based on a fork of the Network Attack Simulator by Jonathon Schwartz (https://github.com/Jjschwartz/NetworkAttackSimulator).

## Citation
Please cite the work using the following bibtex.

```
@inproceedings{10.1145/3560828.3564006,
author = {Reti, Daniel and Fraunholz, Daniel and Elzer, Karina and Schneider, Daniel and Schotten, Hans Dieter},
title = {Evaluating Deception and Moving Target Defense with Network Attack Simulation},
year = {2022},
isbn = {9781450398787},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3560828.3564006},
doi = {10.1145/3560828.3564006},
abstract = {In the field of network security, with the ongoing arms race between attackers, seeking new vulnerabilities to bypass defense mechanisms and defenders reinforcing their prevention, detection and response strategies, the novel concept of cyber deception has emerged. Starting from the well-known example of honeypots, many other deception strategies have been developed such as honeytokens and moving target defense, all sharing the objective of creating uncertainty for attackers and increasing the chance for the attacker of making mistakes. In this paper a methodology to evaluate the effectiveness of honeypots and moving target defense in a network is presented. This methodology allows to quantitatively measure the effectiveness in a simulation environment, allowing to make recommendations on how many honeypots to deploy and on how quickly network addresses have to be mutated to effectively disrupt an attack in multiple network and attacker configurations. With this optimum, attacks can be detected and slowed down with a minimal resource and configuration overhead. With the provided methodology, the optimal number of honeypots to be deployed and the optimal network address mutation interval can be determined. Furthermore, this work provides guidance on how to optimally deploy and configure them with respect to the attacker model and several network parameters.},
booktitle = {Proceedings of the 9th ACM Workshop on Moving Target Defense},
pages = {45–53},
numpages = {9},
keywords = {cyber deception, network security, honeypot, moving target defense},
location = {Los Angeles, CA, USA},
series = {MTD'22}
}
```