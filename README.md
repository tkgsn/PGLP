# Policy Graph-based Location Privacy (PGLP)

Technical Report: Customizable and Rigorous Location Privacy through Policy Graph

https://www.db.soc.i.kyoto-u.ac.jp/~cao/pglp.pdf


Location privacy has been extensively studied in the literature.
However, existing location privacy models are either not rigorous or not customizable, which limits the trade-off between privacy and utility in many real-world applications.
To address this issue, we propose a new location privacy notion called **Policy Graph-based Location Privacy (PGLP)**,  providing a rich interface to release private locations with customizable and rigorous privacy guarantee.

Our contributions are three-folded.
First, the privacy metrics of PGLP is formally designed by extending differential privacy.
Specifically, we formalize a user's location privacy requirements using **location policy graph**, which is expressive and customizable.
Second, we investigate how to satisfy an arbitrarily given location policy graph under adversarial knowledge.
We find that a location policy graph may not always be viable and may suffer \textit{location exposure} when the attacker knows the user's mobility pattern.
We propose efficient methods to detect location exposure and repair the policy graph with optimal utility.
Third, we design a private location trace release framework that pipelines the detection of location exposure,  policy graph repair, and private trajectory release with customizable and rigorous location privacy.
Finally,  we conduct experiments on real-world datasets to verify the effectiveness of the privacy-utility trade-off and the efficiency of the proposed algorithms.

## Implementation
This code was used in the experiments of the above paper.
This is implemented by Python 3.7.


## Reference

- [VLDB 2020 demo] PANDA: Policy-aware Location Privacy for Epidemic Surveillance.
Yang Cao, Shun Takagi, Yonghui Xiao, Li Xiong, Masatoshi Yoshikawa




