# Policy Graph-based Location Privacy (PGLP)


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


## References

- [**ESORICS 2020**] PGLP: Customizable and Rigorous Location Privacy through Policy Graph. <br>
Yang Cao, Yonghui Xiao, Shun Takagi, Li Xiong, Masatoshi Yoshikawa, Yilin Shen, Jinfei Liu, Hongxia Jin, Xiaofeng Xu. <br>
https://arxiv.org/abs/2005.01263

- [**VLDB 2020 demo**] PANDA: Policy-aware Location Privacy for Epidemic Surveillance. <br>
Yang Cao, Shun Takagi, Yonghui Xiao, Li Xiong, Masatoshi Yoshikawa. <br>
https://arxiv.org/abs/2005.00186

- A Demo of PGLP for Epidemiological Analysis: https://github.com/tkgsn/covid-demo



## Contributors

Shun Takagi (Kyoto University) takagi.shun.45a@st.kyoto-u.ac.jp 

Yang Cao (Kyoto University) yang@i.kyoto-u.ac.jp



