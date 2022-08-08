# qc-ga
We are using DEAP package to initiate and continue our evolution, and we are using ProjectQ to perform quantum computer simulations. 
Currently, we are using a set of genetic operators and selection procedure highly similar to [Potocek thing], but we allowed user to describe the topology of the quantum computer the proposed algorithms may be run on. 
We are still working on the project and we are planning to improve it further.

More details will be added later on.

TODO: Add link to paper

## Requirements

Python version: 3.9 (3.10 does not work due to numpy 1.19 requirement)

Libraries:
- libhdf5 (required by qclib)

Pip modules (defined in [requirements.txt](./requirements.txt)): TODO

## Usage

## Roadmap

## Contributing

## TODOS:
- Save statevector (and noisy densitymatrix) of circuit outcome
- check remove consecutive inverse gates
- create picture of ga process and data processing

## Known issues

- does not filter empty/trivial circuits and gates
- does not combine e.g. sqrtx gates

## Authors and acknowledgements
Authors who contributed directly to the codebase and research:
- Berat Yenilen was the primary developer and implemented the genetic algorithm
- Tom Rindell helped with developement and data analysis
- Niklas Halonen helped with project cleanup and data analysis
- Matti Raasakka led the project and did theoretical research

Services and projects we wish to acknowledge
- [Aalto Scientific computing](https://scicomp.aalto.fi/about/) for the Triton cluster
- The authors wish to acknowledge [CSC](https://www.csc.fi/en/home) – IT Center for Science, Finland, for computational resources
- [DEAP](https://github.com/deap/deap)
- [ProjectQ](https://github.com/projectq-framework)
- [Qiskit](https://github.com/Qiskit/qiskit)

## Project status
The project is in its final stages and will see no future updates. 

## License

```
Copyright 2022 Aalto University, Berat Yenilen, Tom Rindell, Niklas Halonen, Matti Raasakka

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```