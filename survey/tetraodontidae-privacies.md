# Tetraodontidae privacies

## Pufferfish privacy
Pufferfish: A Framework for Mathematical Privacy Definitions, DANIEL KIFER, ASHWIN MACHANAVAJJHALA, ACM Transactions on Database Systems

###  Summary
- Bayesian privacy framework
- privacy definition that are customized to the needs of a given application
- Pufferfish defines adversarial knowledge using a set of data generating distributions

**Goal**
- to allow experts in an application domain, who frequently do not have expertise in privacy, to develop rigorous privacy definitions for their data sharing needs

### Definition
A domain expert specifies following three crucial components:
- a set of potential secrets $\mathbb{S}$
- a set of discriminative pairs $\mathbb{S}_{pairs}\subseteq\mathbb{S}\times\mathbb{S}$
- a collection of data evolution scenarios $\mathbb{D}$

**$\mathbb{S}$**
- what we want to protected
e.g., "the regord for individual $h_i$ is not in the data"
- provide a domain for the discriminative pairs

**$\mathbb{S}_{pairs}$**
- a subset of $\mathbb{S}\times\mathbb{S}$
- tell us how to protect the potential secrets
- if $(s_i, s_j)\in\mathbb{S}_{pairs}$, attackers are unable to distinguish between the case where $s_i$ is true of the crutial data and the case where $s_j$ is true of the actual data
- $s_i$ and $s_j$ must be mutually exclusive but not necessarily exhaustive
e.g., ("Bob is in the table", "Bob is not in the table")

**$\mathbb{D}$**
- intuitively, a set of conservative assumptions about how the data evolved and about knowledge of potential attackers
- formaly, a set of probability distributions over $\mathcal{I}$ (the possible database instancces)



### Pufferfish Privacy
- Semantic Guarantee
$\mathrm{e}^{-\epsilon}\leq$ odds ratio $\leq\mathrm{e}^{\epsilon}$
where odds ratio is the ratio of prior odd and posterior odd of secrets.


## Blowfish privacy
Blowfish Privacy: Tuning Privacy-Utility Trade-offs using Policies, Xi He, SIGMOD2014

### Summary
- inspired from Pufferfish, and Blowfish is equivalent to specific instantiations of semantic definitions arising from the Pufferefish framework
- provide a richer set of "tuning knobs"
- privacy policy that specifies two more paramters to permit more utility
	- which information must be kept secret about individuals
	- what constraints may be known publicly about the data


### Policy
- triple $P=(\mathcal{T}, G, \mathcal{I}_Q)$
- $G=(V,E)$ is a discriminative secret graph with $V\subseteq\mathcal{T}$
- $\mathcal{I}$ is a domain of dataset and Q is a constraint, $\mathcal{I}_Q$ is constrained domain of $\mathcal{I}$ by Q.

### Neighbors
- $D_1$ and $D_2$ are neighbors in $P$ if ...
	- $D_1$ and $D_2$ satisfy constraint (= in $\mathcal{I}_Q$)
	- $T\neq \phi$ (= there exists at least an $i$th record such that $D_1 [i]$ and $D_2 [i]$ are connected in $G$)
	- $D_1$ and $D_2$ have the minimum relationship

### Definition
- For neighboring datasets in $P$, Globefish guarantees the indistinguishability


## Globefish privacy
Instantiation of Blowfish to the location setting

### Policy
- $\mathcal{I}$ = $\mathcal{T}$ is a domain of a location
- $\mathcal{I}_Q$ is a constrained domain of a location
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MDU4NDk3LC02NjE2NTA3NTgsLTE5Nj
IzODQzMjYsLTE0MTIwNTY0OTUsLTYzMzU0OTM0NSw3OTA4NTE5
MDUsNTg4MjA1MjU1LDg2NjQ4MDA2NiwtMjExMjMzOTkzOCwtMT
E1MzAxMDU1MCwtMjAwMTg3MDg0MCwtODE2NjgyNzEsLTE4OTIz
ODMzNDUsMjcxNTAzMjMxLDg1ODA3NTMyNywtODgzNzI3Mjc1LD
czMDk5ODExNl19
-->