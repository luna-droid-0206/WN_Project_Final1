**LITERATURE SURVEY**

Attention-Based Deep Learning for Adaptive Modulation and

Coding Scheme (MCS) Selection in Dynamic Wireless Environments

**Team Members**

CS23B1081 A Jeevan Pratheek

CS23B1067 A Hari Hara

CS23B1070 K Yaswanth

CS23B1021 V Siddartha

Course: Wireless Networks

**1. Introduction**

The rapid evolution of wireless communication systems --- from 4G LTE to
5G New Radio (NR) and beyond --- has intensified the demand for
intelligent, adaptive physical-layer mechanisms. At the heart of
reliable wireless transmission lies the process of link adaptation,
which dynamically selects the most suitable Modulation and Coding Scheme
(MCS) based on prevailing channel conditions. Traditional approaches
rely on predetermined lookup tables driven by Channel Quality Indicators
(CQI), which lack the flexibility to respond to highly dynamic and
heterogeneous environments.

The emergence of deep learning and, more recently, attention-based
neural architectures has opened new avenues for replacing rigid
rule-based systems with data-driven models capable of learning complex
temporal and spatial channel patterns. This literature survey reviews
ten significant research contributions from reputed venues --- IEEE
Transactions, IEEE Access, Springer, and Elsevier --- published between
2019 and 2025, that collectively motivate and contextualize the proposed
project: an attention-based deep learning framework for adaptive MCS
selection.

The surveyed papers span multiple paradigms --- deep reinforcement
learning (DRL), convolutional and recurrent neural networks, generative
adversarial networks, Transformer-based architectures, and comprehensive
ML surveys. Together, they paint a coherent picture of how the field has
evolved and where the most promising open research problems lie.

**2. Background and Motivation**

**2.1 Traditional Link Adaptation and Its Limitations**

Conventional link adaptation in cellular systems operates through Outer
Loop Link Adaptation (OLLA) and CQI-based MCS tables. While
computationally simple, these methods struggle under fast-fading
channels, multi-user interference, and scenarios with outdated channel
state information (CSI). The CQI feedback mechanism, which forms the
backbone of LTE and 5G NR link adaptation, introduces inherent latency
that becomes increasingly detrimental as channel coherence time shrinks
at higher mobility or carrier frequencies.

The traditional OLLA approach uses a simple running offset added to the
CQI to compensate for systematic mismatches between the reported CQI and
the actual block error rate (BLER) experienced. While this offset adapts
over time, it is fundamentally reactive and cannot anticipate sudden
channel changes or capture the complex multivariate dependencies between
channel statistics and optimal MCS. Furthermore, standard CQI
quantization (4-bit in LTE, wideband) discards fine-grained spectral
information that could otherwise guide more precise MCS decisions.

**2.2 The Case for Machine Learning in Link Adaptation**

Machine learning approaches offer a fundamentally different paradigm:
instead of reacting to instantaneous measurements with fixed rules, ML
models learn statistical mappings between observable channel features
and optimal transmission parameters. These models can capture nonlinear
relationships, exploit temporal correlations, and generalize across
diverse channel conditions --- capabilities that are simply unavailable
to rule-based systems.

The research landscape in ML-based MCS selection has evolved through
several distinct phases: (1) classical supervised learning using shallow
classifiers such as Support Vector Machines (SVM) and Random Forests,
(2) deep supervised learning using DNNs, CNNs, and LSTMs, (3) deep
reinforcement learning that frames MCS selection as a sequential
decision problem, and (4) emerging attention-based models that
selectively weight relevant features in the channel representation. This
survey traces this trajectory through ten key papers.

**3. Detailed Review of Surveyed Papers**

**Paper 1: Deep Reinforcement Learning-Based Modulation and Coding
Scheme Selection in Cognitive Heterogeneous Networks**

  ---------------- --------------------------------------------------------------
  **Authors**      Q. Zhang, M. C. Gursoy, S. Velipasalar
  **Venue**        IEEE Transactions on Wireless Communications, Vol. 18, No. 6
  **Year**         2019
  **DOI**          https://ieeexplore.ieee.org/document/8703432
  **Method**       Deep Reinforcement Learning (DQN / Policy Gradient)
  **Key Metric**   Throughput (90--100% of optimal), Spectrum Utilization
  ---------------- --------------------------------------------------------------

**3.1.1 Problem Context and Motivation**

Zhang et al. address the MCS selection problem in cognitive
heterogeneous networks (HetNets), where secondary users (SUs) must
simultaneously maximize their own throughput and avoid causing harmful
interference to primary users (PUs). This dual-objective setting makes
MCS selection significantly more complex than in traditional single-user
scenarios, as the optimal choice depends not only on the SU\'s own
channel quality but also on the activity patterns of PUs and the
interference they could cause.

Conventional approaches to this problem typically rely on either
predefined thresholds (which fail to adapt to dynamic PU behavior) or
online optimization methods (which incur prohibitive computational
overhead at each transmission slot). The authors argue that deep
reinforcement learning is uniquely suited to this setting because it can
learn a near-optimal policy through interaction with the environment
without requiring an explicit model of the spectrum dynamics or
interference structure.

**3.1.2 Methodology and Architecture**

The proposed framework models the MCS selection problem as a Markov
Decision Process (MDP). The state space is defined by the observed
received signal strength indicators (RSSI) from both SU and PU
transmissions, the current buffer occupancy, and the history of selected
MCS indices. The action space consists of the available MCS options
(modulation order and code rate combinations). The reward function is
designed to simultaneously maximize SU throughput while penalizing
interference events above a defined threshold.

The DRL agent employs a Deep Q-Network (DQN) with experience replay and
target network stabilization. The Q-network is a fully connected neural
network with three hidden layers, taking the state vector as input and
outputting Q-values for each possible MCS action. During training, the
agent uses an ε-greedy exploration policy with decaying ε to balance
exploration of new MCS choices against exploitation of learned
knowledge. A key architectural contribution is the incorporation of Long
Short-Term Memory (LSTM) units in the Q-network, allowing the agent to
reason about temporal patterns in PU activity rather than relying solely
on the current snapshot.

**3.1.3 Experimental Setup and Results**

Experiments are conducted in a simulated HetNet environment with one
primary user pair and one secondary user pair operating over an
overlapping spectrum band. Channel conditions follow a Rayleigh fading
model with path loss and log-normal shadowing. The agent is trained over
100,000 episodes and evaluated across SNR ranges of 0--30 dB and under
varying PU activity fractions (10%--90% channel occupancy).

The DRL agent achieves 90--100% of the throughput attained by an oracle
policy with perfect channel state information across all tested SNR
regimes. Compared to a fixed-MCS baseline, the DRL approach delivers up
to 40% throughput improvement at medium SNR. Interference events (cases
where the SU transmits on an occupied channel) are reduced by over 60%
compared to the greedy MCS selection baseline.

**3.1.4 Relevance to the Proposed Project**

This paper establishes a performance ceiling for data-driven MCS
selection that any supervised or attention-based approach should aspire
to match. It also highlights the importance of temporal context in MCS
decisions --- a motivation directly addressed by attention mechanisms
that can model long-range dependencies in historical channel
observations. The DRL framework, while achieving near-optimal
performance, has a significant drawback: it requires extensive online
training time and is difficult to deploy in real-time physical layer
systems. The proposed attention-based model aims to achieve comparable
performance with an offline-trained, fast-inference architecture.

**Paper 2: Adaptive Modulation and Coding Using Deep Recurrent Neural
Network**

  ---------------- --------------------------------------------------------------
  **Authors**      A. Mohammadvaliei, M. Ardebilipour, H. R. Bahrami
  **Venue**        Telecommunication Systems, Springer, Vol. 81
  **Year**         2022
  **DOI**          https://link.springer.com/article/10.1007/s11235-022-00965-4
  **Method**       1D CNN + LSTM (Dual Architecture)
  **Key Metric**   Bit Error Rate (BER), Spectral Efficiency, Throughput
  ---------------- --------------------------------------------------------------

**3.2.1 Problem Context and Motivation**

Mohammadvaliei et al. focus on the problem of adaptive modulation and
coding in OFDM (Orthogonal Frequency Division Multiplexing) systems,
which are the modulation scheme of choice in LTE, 5G NR, and Wi-Fi
standards. OFDM\'s strength --- parallel transmission across multiple
subcarriers --- also introduces complexity in link adaptation because
channel quality can vary significantly across subcarriers
(frequency-selective fading). Standard CQI-based approaches use a single
wideband CQI or a small number of subband CQIs, which fails to capture
fine-grained per-subcarrier variation.

The authors hypothesize that a combination of convolutional feature
extraction (to capture local spectral patterns across subcarriers) and
recurrent sequence modeling (to capture temporal channel dynamics across
OFDM symbols) can outperform single-modality architectures for AMC in
frequency-selective fading channels.

**3.2.2 Methodology and Architecture**

The proposed architecture consists of two stages. In the first stage, a
1D Convolutional Neural Network processes the complex channel frequency
response (CFR) across OFDM subcarriers, applying multiple filter banks
to extract local spectral patterns such as channel magnitude variations,
phase offsets, and coherence bandwidth features. These convolutional
features are computed independently for each OFDM symbol in a sliding
window.

In the second stage, the sequence of convolutional feature vectors
across consecutive OFDM symbols is fed into an LSTM network. The LSTM\'s
recurrent connections allow it to model temporal dependencies in the
channel --- capturing how channel coefficients evolve over time due to
Doppler spread and multipath dynamics. The LSTM output at each time step
is connected to a softmax classifier that predicts the optimal MCS index
from a predefined set of 15 MCS options (matching the 4G LTE CQI table).

The model is trained end-to-end using cross-entropy loss over labeled
training samples, where labels are generated by an oracle that selects
the highest-rate MCS satisfying a BER constraint of 10⁻³. Training uses
the Adam optimizer with a learning rate schedule and batch normalization
between convolutional layers.

**3.2.3 Experimental Setup and Results**

Experiments are conducted under the ITU Vehicular A channel model
(simulating high-mobility vehicular scenarios) and the ETU (Extended
Typical Urban) model at carrier frequencies of 2.1 GHz. OFDM parameters
follow the LTE numerology: 15 kHz subcarrier spacing, 12 subcarriers per
resource block. The train/test split uses SNR-stratified sampling to
ensure coverage across 0--30 dB SNR.

The joint CNN-LSTM model achieves a 2.3 dB gain in SNR efficiency
compared to standalone LSTM (no CNN) and a 3.1 dB gain compared to a
standard lookup-table approach at the 10⁻³ BER threshold. Spectral
efficiency is improved by 18% at 15 dB SNR relative to the CQI-based
baseline. The authors also demonstrate that the CNN preprocessing
significantly reduces the LSTM input dimensionality, leading to 35%
faster training convergence.

**3.2.4 Relevance to the Proposed Project**

This paper serves as a direct baseline for the proposed attention-based
model. The CNN-LSTM architecture captures both local spectral patterns
and temporal dynamics, making it a strong competitor. However, LSTMs
suffer from a fundamental limitation: they process sequences strictly
sequentially, which (1) limits parallelization during training, (2)
makes it difficult to capture very long-range dependencies due to
vanishing gradients, and (3) cannot selectively attend to the most
informative OFDM symbols in a window. The attention mechanism proposed
in our project directly addresses all three limitations, offering a more
expressive and efficiently trainable alternative.

**Paper 3: Machine Learning-Based Methods for MCS Prediction in 5G
Networks**

  ---------------- --------------------------------------------------------------
  **Authors**      A. Hammoodi, I. Al-Hamdani, S. Al-Rubaye
  **Venue**        Telecommunication Systems, Springer, 2024
  **DOI**          https://link.springer.com/article/10.1007/s11235-024-01158-x
  **Method**       Deep Neural Network (DNN), SVM, Random Forest (Comparative)
  **Key Metric**   MCS Prediction Accuracy, Throughput, Inference Latency
  ---------------- --------------------------------------------------------------

**3.3.1 Problem Context and Motivation**

Hammoodi et al. present the most comprehensive comparative study of
machine learning techniques for MCS prediction in OFDM-based 5G networks
available in the literature. Unlike prior works that propose and
evaluate a single architecture, this paper systematically benchmarks
multiple ML families --- traditional classifiers and deep networks ---
under a unified experimental framework, enabling fair and direct
comparison.

The authors are motivated by a fundamental question in the field: given
the growing zoo of ML approaches for wireless link adaptation, which
family of models offers the best trade-off between prediction accuracy,
generalization across channel conditions, and real-time inference
feasibility? This question has practical importance because different
deployment scenarios (handset, gNB, edge server) impose different
computational constraints.

**3.3.2 Methodology and Architecture**

The paper evaluates five distinct ML methods for MCS prediction: (1)
Support Vector Machine with RBF kernel (SVM-RBF), (2) Random Forest with
100 trees, (3) Gradient Boosting (XGBoost), (4) a standard fully
connected Deep Neural Network with 4 hidden layers and ReLU activations,
and (5) a specialized DNN with residual skip connections for improved
gradient flow.

The input feature vector for all models consists of: wideband SNR, CQI
report, rank indicator (MIMO rank), precoding matrix indicator, RSRP,
RSRQ, and SINR measurements, along with a 10-symbol history of previous
MCS selections. This feature engineering approach is notable because it
incorporates domain knowledge --- selecting measurements that are
already available at the gNB scheduler --- making the prediction
pipeline directly deployable without additional sensing infrastructure.

The MCS prediction is framed as a 29-class classification problem (MCS
indices 0--28 in the 5G NR MCS table for 64-QAM). The models are trained
on a dataset of 500,000 samples collected from a 5G NR system-level
simulator, with 80/20 train/test split stratified by SNR range.

**3.3.3 Experimental Setup and Results**

Experiments cover three channel scenarios: AWGN (static baseline), TDL-C
(Tapped Delay Line, high-delay-spread, representative of dense urban),
and CDL-D (Clustered Delay Line, LOS-dominant, suburban). Speed
conditions range from 3 km/h (pedestrian) to 120 km/h (vehicular).

The residual DNN achieves the highest overall accuracy at 94.3% correct
MCS prediction across all scenarios, outperforming SVM (87.1%), Random
Forest (89.6%), and XGBoost (91.2%). The performance gap between deep
and shallow methods widens under the TDL-C channel at 120 km/h, where
the residual DNN maintains 91.8% accuracy while SVM drops to 79.3%. The
authors attribute this to the DNN\'s ability to learn nonlinear feature
interactions that become more complex under high-mobility channels.

Notably, all ML methods outperform the CQI-based lookup table baseline
by 8--15% in throughput, confirming the general superiority of learned
approaches. The residual DNN also demonstrates the fastest inference at
0.23 ms on a GPU server, meeting 5G NR\'s 1 ms HARQ feedback deadline.

**3.3.4 Relevance to the Proposed Project**

This paper provides the most direct baseline for the proposed project,
as it establishes the state-of-the-art accuracy for DNN-based MCS
prediction in 5G NR. The residual DNN\'s 94.3% accuracy sets a concrete
target for the attention-based model to meet or exceed. Furthermore, the
paper\'s feature engineering approach --- using standard gNB
measurements as inputs --- provides a practical blueprint for
constructing the input representation of the proposed model. The
identification of performance degradation at high mobility further
motivates the attention mechanism, which can selectively weight more
recent channel observations when mobility is high.

**Paper 4: Joint Power Allocation and MCS Selection for Energy-Efficient
Link Adaptation: A Deep RL Approach**

  ---------------- -------------------------------------------------------------------------
  **Venue**        Computer Networks, Elsevier, Vol. 217
  **Year**         2022
  **DOI**          https://www.sciencedirect.com/science/article/abs/pii/S1389128622004200
  **Method**       Deep Reinforcement Learning (Actor-Critic / PPO)
  **Key Metric**   Energy Efficiency (bits/Joule), Throughput-Power Trade-off
  ---------------- -------------------------------------------------------------------------

**3.4.1 Problem Context and Motivation**

This Elsevier paper extends the MCS selection problem by jointly
optimizing MCS and power allocation under an energy efficiency
criterion. In conventional approaches, MCS selection and power control
are handled by separate algorithms that do not account for their
coupling: a higher modulation order requires higher SNR (and thus higher
transmit power), while a lower code rate improves robustness but reduces
spectral efficiency. Decoupled optimization of these two variables is
provably suboptimal, yet coordinated optimization is combinatorially
complex.

The paper frames the joint problem as a Markov Decision Process where
both the MCS index and the transmit power level are included in the
action space. The reward function combines throughput (in bits per
resource block) with a power cost term, creating an explicit energy
efficiency objective. This formulation is particularly relevant for
uplink scenarios in IoT or mobile devices where battery life is a
critical constraint.

**3.4.2 Methodology and Architecture**

The DRL agent uses a Proximal Policy Optimization (PPO) algorithm with
an actor-critic neural network architecture. The actor network outputs a
joint action: a probability distribution over MCS indices (discrete) and
a continuous transmit power value (which is discretized into 10 power
levels for tractability). The critic network estimates the expected
cumulative reward (value function) given the current state.

The state vector includes: instantaneous SNR measured at the receiver,
the previous MCS and power level selections, the current buffer status,
and an exponentially weighted moving average of recent SINR values to
capture channel trend. The policy network has two hidden layers with 256
neurons each and uses orthogonal initialization for stable training. A
clipping parameter ε = 0.2 is used in the PPO objective to limit policy
update steps, preventing divergence during early training.

A key contribution is the curriculum learning strategy: the agent is
initially trained in AWGN conditions to learn basic MCS-to-SNR mappings,
then progressively exposed to Rayleigh fading channels with increasing
Doppler spread. This staged training significantly accelerates
convergence compared to training directly on the most challenging
channel conditions.

**3.4.3 Experimental Setup and Results**

Simulations are conducted in a single-cell downlink scenario with one
base station and one user equipment, using an LTE-like physical layer
with 50 resource blocks. Channel models include AWGN, Rayleigh flat
fading, and frequency-selective TDL-C. Energy efficiency is measured in
bits per Joule, assuming a linear power amplifier model where transmit
power directly maps to energy consumption.

The joint DRL agent achieves 15--22% improvement in energy efficiency
compared to separate (decoupled) optimization of MCS and power. At
medium SNR (15 dB), energy efficiency gains reach 31% over a fixed-MCS,
full-power baseline. The curriculum learning approach reduces training
time by 40% compared to direct training, confirming its effectiveness.

**3.4.4 Relevance to the Proposed Project**

This paper highlights the interdependency between MCS selection and
power control, suggesting that future attention-based models could
benefit from multi-task learning formulations that simultaneously
address both variables. The energy efficiency perspective is also
directly relevant to 5G IoT and mMTC (massive Machine-Type
Communications) scenarios where the proposed attention model could be
deployed. The PPO approach used here provides an alternative
optimization strategy that could be explored in future extensions of the
proposed project.

**Paper 5: Adaptive Modulation and Coding in 5G Networks with Deep
Learning**

  ---------------- -----------------------------------------------------------------
  **Venue**        IEEE Xplore Conference Publication, 2024
  **DOI**          https://ieeexplore.ieee.org/iel8/10763531/10763546/10763790.pdf
  **Method**       CNN + RNN (GRU) with QoS-Aware Training
  **Key Metric**   Throughput, QoS Satisfaction Rate, BER
  ---------------- -----------------------------------------------------------------

**3.5.1 Problem Context and Motivation**

This 2024 IEEE conference paper addresses adaptive modulation and coding
in 5G NR networks with an explicit focus on Quality of Service (QoS)
constraints. Unlike prior works that optimize purely for throughput or
BER in isolation, this paper introduces a QoS-constrained learning
objective: the model must select the highest-rate MCS that satisfies
user-specific BER targets, which may differ across applications (e.g.,
\< 10⁻³ for voice, \< 10⁻⁵ for critical data). This more realistic
formulation acknowledges that different services have different
reliability requirements.

The work is motivated by the observation that standard deep learning MCS
predictors trained to maximize throughput often violate BER constraints
for a non-trivial fraction of transmissions, leading to excessive HARQ
retransmissions that erode the throughput gains. Introducing explicit
QoS constraints into the learning objective directly addresses this
failure mode.

**3.5.2 Methodology and Architecture**

The proposed model uses a hybrid CNN-GRU (Gated Recurrent Unit)
architecture. The CNN module processes a 2D channel feature map --- a
matrix of complex channel coefficients across time (OFDM symbols) and
frequency (subcarriers) --- using 2D convolutional filters that extract
joint time-frequency patterns. This 2D convolution captures
frequency-selective fading patterns that are invisible to per-subcarrier
1D processing.

The extracted CNN features are then fed into a GRU layer, which models
temporal evolution of the channel. GRUs are chosen over LSTMs for their
smaller parameter count and comparable temporal modeling capability,
which is advantageous for deployment on resource-constrained hardware
such as FPGAs in a base station.

The key methodological innovation is the QoS-aware loss function. In
addition to the standard cross-entropy classification loss for MCS index
prediction, the training objective includes a penalty term that fires
whenever the predicted MCS results in a BER exceeding the target
threshold (computed using a pre-trained BER-vs-SNR lookup model for each
MCS). This penalty is differentiable and can be backpropagated through
the network during training.

**3.5.3 Experimental Setup and Results**

The model is evaluated under 5G NR channel models (CDL-A for NLOS and
CDL-D for LOS) at sub-6 GHz frequencies. UE speeds range from 3 to 120
km/h. QoS targets are set at 10⁻³ BER for best-effort data and 10⁻⁵ for
ultra-reliable communications. Three service classes are evaluated to
test the model\'s ability to adapt to heterogeneous QoS requirements.

The QoS-aware CNN-GRU model achieves 97.2% QoS satisfaction rate
(fraction of transmissions meeting the BER target) while maintaining 92%
of the maximum achievable throughput. In comparison, a
throughput-maximizing DNN without QoS constraints achieves only 78% QoS
satisfaction despite higher raw throughput. The model demonstrates
consistent performance across UE speeds, confirming that the GRU
temporal modeling effectively handles the increased Doppler spread at
120 km/h.

**3.5.4 Relevance to the Proposed Project**

The QoS-aware training paradigm introduced in this paper provides a
practical framework for deploying the proposed attention-based model in
real 5G NR environments. The concept of incorporating
application-specific reliability constraints into the learning objective
is directly transferable to an attention-based architecture.
Furthermore, the 2D convolutional feature extraction approach ---
treating the channel matrix as a time-frequency image --- suggests a
promising input representation for the attention model, where multi-head
self-attention could replace or augment the 2D convolution to capture
both local and global time-frequency dependencies.

**Paper 6: GAN-Based Adaptive Modulation and Coding for Next-Generation
5G Communication Systems**

  ---------------- --------------------------------------------------------------
  **Venue**        Discover Applied Sciences, Springer, 2025
  **DOI**          https://link.springer.com/article/10.1007/s42452-025-06509-0
  **Method**       Generative Adversarial Network (GAN) + Classifier
  **Key Metric**   Throughput, Reliability, Data Augmentation Quality
  ---------------- --------------------------------------------------------------

**3.6.1 Problem Context and Motivation**

A persistent challenge in applying supervised ML to wireless link
adaptation is the scarcity of labeled training data for rare but
critical channel conditions. In practice, a deployed 5G network may
rarely experience extreme scenarios (e.g., very high Doppler, deep
fades, severe interference) during the data collection phase, resulting
in under-represented samples for these conditions in the training
dataset. A classifier trained on such an imbalanced dataset will be
biased toward common channel conditions and may fail precisely when
reliable performance is most needed.

This 2025 Springer paper proposes using Generative Adversarial Networks
(GANs) to generate realistic synthetic channel samples for rare and
extreme conditions, augmenting the training dataset and improving
classifier robustness across the full range of channel environments. The
core insight is that a GAN trained on observed channel data can capture
the multivariate statistical structure of the channel (including
correlations between subcarriers, temporal evolution, and rare event
statistics) and generate diverse, realistic samples on demand.

**3.6.2 Methodology and Architecture**

The system consists of two components: a Conditional GAN (cGAN) for
channel data augmentation and a CNN-based MCS classifier trained on the
augmented dataset. The cGAN\'s generator takes a noise vector and a
condition vector (desired SNR range, Doppler class, delay spread class)
as inputs and outputs synthetic complex channel coefficients across
subcarriers and time slots. The discriminator attempts to distinguish
real channel measurements from generated ones, conditioning on the same
condition vector.

The cGAN is trained using the Wasserstein GAN with Gradient Penalty
(WGAN-GP) objective, which provides more stable training dynamics than
the original GAN formulation and avoids mode collapse --- a common
failure mode where the generator produces a limited variety of samples.
Training uses a two-phase protocol: first, the cGAN is pre-trained on
all available real channel data; second, the cGAN is fine-tuned on
target rare conditions using few-shot learning with only 100--500 real
samples per rare class.

The augmented dataset (80% real, 20% synthetic) is used to train the
CNN-based MCS classifier. The classifier architecture is standard: three
convolutional layers followed by global average pooling and a fully
connected softmax output layer. No architectural changes are made to the
classifier itself; the innovation lies entirely in the training data
augmentation strategy.

**3.6.3 Experimental Setup and Results**

The GAN-based approach is evaluated in two regimes: (1) a dataset with
natural class imbalance (reflecting a real deployment where extreme
channels are rare) and (2) a balanced synthetic dataset generated
entirely by the cGAN. The classifier is evaluated on real channel
samples from a held-out test set that includes high proportions of rare
channel conditions.

GAN augmentation improves MCS classification accuracy from 87.4% (no
augmentation) to 93.6% on rare channel conditions --- a gain of 6.2
percentage points. On common channel conditions, accuracy remains stable
at 95.1%. The WGAN-GP training converges in approximately 15,000
iterations, and the generated samples pass a Fréchet Inception Distance
(FID)-inspired metric adapted for wireless channel statistics.
End-to-end system throughput improves by 8.7% over the no-augmentation
baseline under the imbalanced dataset scenario.

**3.6.4 Relevance to the Proposed Project**

The GAN-based augmentation framework addresses a real and practical
challenge for deploying the proposed attention-based model in production
5G networks. Rather than replacing the attention model, the cGAN could
serve as a data augmentation preprocessing step that ensures balanced
representation of all channel conditions in the training data. This is
particularly relevant because the attention mechanism\'s ability to
selectively focus on rare channel patterns depends on adequate exposure
to such patterns during training. An alternative perspective offered by
this paper is that the GAN\'s learned channel representations could
potentially serve as input features for the attention model, providing
richer statistical context than raw channel coefficients.

**Paper 7: Emerging Tools for Link Adaptation on 5G NR and Beyond ---
Challenges and Opportunities**

  ---------------- ------------------------------------------------------------------------
  **Authors**      F. J. Martín-Vega, F. J. López-Martínez, G. Gomez, M. C. Aguayo-Torres
  **Venue**        IEEE Access, Vol. 9
  **Year**         2021
  **DOI**          https://doi.org/10.1109/ACCESS.2021.3111783
  **Method**       Comprehensive Survey (ML, DRL, Model-based Hybrid)
  **Key Metric**   Spectral Efficiency Gain, BLER Target Achievement, Coverage
  ---------------- ------------------------------------------------------------------------

**3.7.1 Scope and Objectives**

Martín-Vega et al. provide the most comprehensive survey of link
adaptation techniques for 5G NR and beyond available in the literature.
The paper covers a 30-year span of developments, from early OLLA methods
to state-of-the-art DRL and neural-network-based approaches, and
provides a structured taxonomy that situates each contribution within
the broader link adaptation research landscape. This survey is an
essential reference for understanding how the field has evolved and what
open challenges remain.

The paper is organized around three central themes: (1) classical link
adaptation methods and their fundamental limitations in 5G NR scenarios,
(2) machine learning approaches spanning supervised, reinforcement, and
transfer learning, and (3) cross-cutting challenges including channel
non-stationarity, feedback overhead, multi-antenna complexity, and
real-time implementation constraints.

**3.7.2 Key Findings on Classical Methods**

The survey provides a rigorous analysis of Outer Loop Link Adaptation
(OLLA), which remains the most widely deployed link adaptation mechanism
in commercial 5G networks. OLLA adjusts the effective CQI by an additive
offset that is updated based on HARQ feedback: the offset increases when
a NACK (negative acknowledgment) is received and decreases after an ACK.
The survey demonstrates analytically that OLLA converges to the correct
offset only under stationary channel conditions, and its convergence
time (typically 50--200 TTIs) is too slow for fast-varying channels.

The paper also covers inner loop link adaptation (ILLA) and hybrid ARQ
(HARQ) combining strategies, establishing that the maximum throughput
achievable with CQI-based methods is fundamentally limited by CQI
quantization resolution and feedback delay. Under 5G NR\'s minimum
latency of 0.5 ms (one slot), the CQI can become outdated at vehicle
speeds above 30 km/h at 3.5 GHz, directly motivating predictive ML-based
approaches.

**3.7.3 Taxonomy of ML-Based Approaches**

The survey categorizes ML-based link adaptation approaches into four
families. Supervised learning methods (including DNNs and CNNs) directly
map channel measurements to MCS indices using labeled training data;
they achieve high accuracy when training and test distributions match
but generalize poorly to unseen channel conditions. Reinforcement
learning methods (DRL, Q-learning) learn adaptive policies through trial
and error and generalize better but require extensive training time.
Transfer learning methods leverage pre-trained models from related tasks
(e.g., channel estimation) and adapt them to MCS prediction with limited
new data. Hybrid model-based ML methods combine physical channel models
with neural networks, achieving better generalization by incorporating
known channel structure.

The survey identifies attention mechanisms and Transformer architectures
as a \"promising emerging direction\" for wireless link adaptation,
citing their success in channel estimation and modulation classification
as evidence for their potential in MCS selection. This observation
directly validates the research direction of the proposed project.

**3.7.4 Open Challenges and Research Directions**

The survey identifies several open challenges that are directly relevant
to the proposed project: (1) generalization across diverse propagation
environments without retraining, (2) handling of non-stationary channels
where the optimal MCS changes rapidly, (3) practical implementation of
ML models within the 1 ms TTI latency budget of 5G NR, (4)
interpretability of ML decisions for certification and debugging, and
(5) robustness to adversarial inputs and measurement noise. Each of
these challenges is addressed or partially addressed by the proposed
attention-based model.

**3.7.5 Relevance to the Proposed Project**

This survey provides the foundational framework within which the
proposed project is situated. By covering the full history of link
adaptation research and identifying attention mechanisms as a key
research frontier, it validates the scientific novelty and practical
importance of the proposed work. The survey\'s analysis of OLLA
limitations provides quantitative justification for moving beyond
CQI-based approaches, while its taxonomy of ML methods clarifies how the
attention-based model fits within the broader landscape of learned link
adaptation techniques.

**Paper 8: Channelformer --- Attention Based Neural Solution for
Wireless Channel Estimation**

  ---------------- ------------------------------------------------------------
  **Authors**      S. Balevi, A. Doshi, A. Jalal, A. Dimakis, J. G. Andrews
  **Venue**        IEEE Transactions on Wireless Communications, 2023
  **DOI**          https://dl.acm.org/doi/abs/10.1109/TWC.2023.3244484
  **Method**       Transformer (Multi-Head Self-Attention), Encoder-Decoder
  **Key Metric**   Channel Estimation MSE, Parameter Count, Inference Latency
  ---------------- ------------------------------------------------------------

**3.8.1 Problem Context and Motivation**

Channelformer addresses wireless channel estimation --- the problem of
accurately recovering the channel frequency response (CFR) at the
receiver --- using a Transformer-based neural network. While channel
estimation is distinct from MCS selection, it is directly relevant
because accurate CSI is the foundation for any MCS selection algorithm:
the quality of the CSI estimate directly determines the accuracy of MCS
decisions. Channelformer is the most widely cited example of a
Transformer architecture applied to a wireless physical layer problem,
and its design principles transfer directly to MCS selection.

The paper is motivated by the limitations of both classical pilot-aided
estimators (which suffer in doubly-dispersive channels) and prior
DNN-based estimators (which use large fully connected networks that are
difficult to deploy on hardware-constrained receivers). The key insight
is that multi-head self-attention can exploit the correlation structure
of wireless channels --- where channel coefficients at nearby
subcarriers or adjacent OFDM symbols are correlated --- in a learnable,
data-driven way.

**3.8.2 Architecture Design**

Channelformer adopts an encoder-decoder Transformer architecture. The
encoder processes pilot measurements (known symbols inserted at fixed
positions in the time-frequency grid) and produces contextual embeddings
that capture channel correlation across the observed positions. The
decoder uses cross-attention to interpolate channel estimates at
non-pilot positions by attending to the encoder\'s contextual
representations.

A critical design contribution is the learnable positional encoding for
the 2D time-frequency grid. Unlike the sinusoidal positional encodings
used in NLP Transformers, Channelformer\'s positional encodings are
learned during training, allowing the network to adapt its spatial
representations to the specific OFDM numerology (subcarrier spacing,
OFDM symbol duration, pilot pattern) of the target standard.

Post-training, the authors apply attention-head pruning: heads that
contribute minimally to estimation accuracy (identified via
gradient-based importance scores) are removed from the network. This
structured pruning reduces the parameter count by up to 70% with less
than 0.2 dB degradation in estimation accuracy. The resulting
lightweight model meets hardware deployment constraints while
maintaining near-full performance.

**3.8.3 Experimental Setup and Results**

Channelformer is evaluated on the DeepMIMO dataset --- a standardized
ray-tracing-based channel dataset widely used in wireless ML research
--- and on 3GPP CDL/TDL channel models at 2.1 GHz and 28 GHz. The model
is compared against LS (Least Squares) estimation, MMSE estimation, and
the ChannelNet deep learning baseline.

Channelformer outperforms the ChannelNet baseline by 1.8 dB in
normalized MSE at 10 dB SNR. After 70% parameter pruning, it still
outperforms unpruned ChannelNet by 0.9 dB while using 3× fewer
parameters. The model achieves 0.31 ms inference latency on a modern
GPU, compatible with 5G NR\'s 0.5 ms slot duration. The attention
visualization confirms that the model correctly identifies channel
coherence bandwidth and coherence time, focusing attention on correlated
pilot positions.

**3.8.4 Design Principles Transferable to MCS Selection**

Channelformer establishes three key design principles directly
applicable to the proposed MCS selection model: (1) learnable positional
encodings for OFDM time-frequency grids, enabling the model to learn the
correlation structure of the specific channel environment; (2)
attention-driven feature selection, where the model automatically
identifies the most informative channel observations rather than
treating all measurements equally; (3) structured pruning for
deployment-ready lightweight models that meet real-time latency
constraints. The proposed project adapts these principles from channel
estimation to MCS prediction, using multi-head self-attention over a
sliding window of historical channel observations to predict the optimal
MCS index.

**Paper 9: Deep Reinforcement Learning Based Adaptive Modulation With
Outdated CSI**

  ---------------- -----------------------------------------------------------
  **Authors**      M. B. Mashhadi, Q. Yang, D. Gündüz
  **Venue**        IEEE Communications Letters, Vol. 25, No. 10
  **Year**         2021
  **DOI**          https://doi.org/10.1109/LCOMM.2021.3132947
  **Method**       Deep Reinforcement Learning (DQN with LSTM state encoder)
  **Key Metric**   Throughput under CSI delay, Outage probability
  ---------------- -----------------------------------------------------------

**3.9.1 Problem Context and Motivation**

Mashhadi et al. address a practically critical but often overlooked
challenge in wireless link adaptation: the effect of outdated CSI on MCS
selection performance. In real systems, the channel state information
used for MCS selection is always delayed relative to the actual
transmission instant. This delay arises from the round-trip time of CQI
feedback (typically 4--8 ms in LTE), processing time at the scheduler,
and the time between channel estimation and data transmission. In
fast-fading environments --- high mobility, mmWave, or frequency bands
with short coherence time --- this delay means the CSI used for MCS
selection may be significantly decorrelated from the actual channel at
transmission time.

The authors quantify this problem precisely: at 60 km/h vehicle speed at
2.1 GHz, the channel coherence time is approximately 5.7 ms, meaning a
CSI delay of 6 ms (one TTI round trip) leads to a spatial correlation
coefficient of only 0.42 between the measured and actual channel. This
decorrelation causes conventional CQI-based MCS selection to severely
over- or under-estimate the appropriate MCS, leading to either excessive
HARQ retransmissions (over-estimation) or unnecessarily low spectral
efficiency (under-estimation).

**3.9.2 Methodology and Architecture**

The proposed DRL agent learns to compensate for CSI delay by building
predictive internal representations of channel dynamics. The state
representation fed to the DRL agent consists of a sequence of past CSI
measurements (delayed by varying amounts) rather than a single
instantaneous measurement. An LSTM encoder processes this historical
sequence and produces a fixed-size context vector that summarizes the
learned channel trend, which is then used as the state input to the DQN.

The DQN architecture uses three fully connected layers after the LSTM
encoder. The action space consists of MCS indices from the 4G LTE table
(CQI 1--15 mapped to QPSK through 64-QAM with various code rates). The
reward function is the instantaneous achievable rate (Shannon capacity
computed from the actual, not delayed, SNR), which ensures the agent
learns to predict the future channel state rather than optimizing for
the outdated measurement.

A key training technique is the use of a simulated delay buffer: during
training, the agent observes the CSI with a configurable delay parameter
d (ranging from 0 to 3 TTIs). Training with multiple delay values
simultaneously --- using curriculum learning that starts with d=0 and
progressively introduces larger delays --- produces an agent that
generalizes well across delay conditions without requiring retraining.

**3.9.3 Experimental Setup and Results**

Experiments are conducted under TDL-B channel model (high delay spread,
typical of urban environments) at 2.1 GHz and 28 GHz. UE speeds range
from 3 to 120 km/h. CSI delays from 0 to 6 ms are evaluated. The DRL
agent is compared against: (1) a non-adaptive fixed-MCS policy, (2) a
CQI-based lookup with no delay compensation, (3) the Kalman filter-based
CSI prediction baseline, and (4) an oracle policy with perfect future
CSI.

The DRL agent with LSTM state encoder achieves 85--94% of the oracle
throughput across all delay and mobility conditions. At the most
challenging condition (6 ms delay, 120 km/h), it outperforms the Kalman
predictor baseline by 12% in throughput and reduces outage probability
(BER \> 10⁻³) by 35%. The improvement over no-compensation baseline
reaches 28% at high mobility, confirming that temporal modeling is
essential under outdated CSI.

**3.9.4 Relevance to the Proposed Project**

This paper provides the most direct motivation for the temporal
attention mechanism in the proposed project. The LSTM encoder used in
this paper captures channel trends sequentially, but self-attention
offers a superior alternative: it can directly compare any two time
steps in the observation window, identifying the most informative
historical observations regardless of their temporal distance. For
example, if the channel follows a periodic pattern (e.g., due to a
rotating reflector), self-attention can learn to focus on observations
at the appropriate temporal lag rather than weighting recent
observations uniformly. The proposed project incorporates a temporal
self-attention module over a sliding window of past CSI measurements,
directly inspired by the temporal modeling strategy in this paper.

**Paper 10: Machine Learning in Adaptive Modulation --- A Survey and
Challenges**

  ---------------- -----------------------------------------------------
  **Authors**      Y. Tan, J. Zheng, Y. Yu
  **Venue**        IEEE Access, Vol. 9
  **Year**         2021
  **DOI**          https://doi.org/10.1109/ACCESS.2021.3091293
  **Method**       Comprehensive Survey (Supervised, Unsupervised, RL)
  **Key Metric**   Survey Coverage, Gap Analysis, Research Roadmap
  ---------------- -----------------------------------------------------

**3.10.1 Scope and Contribution**

Tan, Zheng, and Yu present a dedicated survey of machine learning
techniques applied specifically to adaptive modulation, providing a
complementary perspective to the broader link adaptation survey by
Martín-Vega et al. (Paper 7). While Paper 7 covers the full link
adaptation pipeline (including OLLA, HARQ, and power control), this
survey focuses exclusively on the modulation order and code rate
selection problem and provides a more granular analysis of ML algorithm
choices within this narrower scope.

The survey covers 85 research papers published between 2012 and 2021,
organized by ML paradigm (supervised, unsupervised, reinforcement) and
further subdivided by neural architecture type (SVM, decision trees,
shallow NNs, deep NNs, CNNs, RNNs, DRL). For each category, the authors
provide a quantitative meta-analysis of reported performance metrics,
enabling direct comparison across papers that used different
experimental setups.

**3.10.2 Supervised Learning Approaches**

The survey\'s analysis of supervised learning for adaptive modulation
covers three subcategories. Shallow classifiers (SVM, k-NN, decision
trees) achieve 80--88% MCS prediction accuracy in static or slowly
varying channels but degrade significantly at high mobility, where the
learned decision boundaries become misaligned with the actual channel
distribution. Feature engineering is identified as the critical factor
for shallow classifier performance, with papers using physics-informed
features (subband SINR, delay spread, Doppler spread) significantly
outperforming those using raw channel coefficients.

Deep neural networks achieve 90--96% accuracy across the surveyed
papers, with the highest-performing architectures using residual
connections and batch normalization. The survey identifies a consistent
pattern: DNNs trained on one channel model (e.g., AWGN) generalize
poorly to others (e.g., Rayleigh fading) when no domain adaptation is
applied. This overfitting to training channel conditions is identified
as the primary limitation of supervised approaches and directly
motivates the use of attention mechanisms, which can learn
channel-agnostic representations.

**3.10.3 Unsupervised and Semi-Supervised Approaches**

The survey covers autoencoders, variational autoencoders (VAEs), and
clustering-based methods for AMC. Autoencoders are used to learn compact
representations of channel conditions in an unsupervised manner, which
are then used as features for a downstream MCS classifier. This approach
achieves 87--91% accuracy with 30--50% fewer labeled training samples
compared to fully supervised methods, making it attractive for scenarios
where labeled data is scarce.

Semi-supervised learning approaches combine small amounts of labeled
data with large amounts of unlabeled channel measurements, using
consistency regularization or pseudo-labeling to leverage unlabeled
samples during training. These methods achieve accuracy within 2--3% of
fully supervised baselines using only 10% of the labeled data,
suggesting strong practical potential for reducing data collection
overhead in deployed networks.

**3.10.4 Reinforcement Learning Approaches**

The survey\'s RL section covers Q-learning, DQL, and actor-critic
methods, analyzing 28 papers. The survey finds that RL methods
consistently outperform supervised approaches in non-stationary
environments where the channel distribution shifts over time (e.g., due
to mobility, network load changes, or interference dynamics). However,
RL methods require 10--100× more computational resources for training
and are significantly harder to implement in production systems due to
the requirement for online environment interaction.

The survey notes a critical gap in the RL-for-AMC literature: almost no
papers evaluate the cost of online training in terms of throughput loss
during the exploration phase, where the agent makes suboptimal MCS
choices to gather experience. This exploration cost can be significant
in high-throughput networks and argues for offline pre-training
strategies or hybrid approaches that combine an offline-trained
supervised model (for safe initial deployment) with online RL
fine-tuning.

**3.10.5 Identified Research Gap --- Attention Mechanisms**

The most significant finding of this survey for the proposed project is
its explicit identification of attention mechanisms as an unexplored but
highly promising direction for adaptive modulation. The authors note
that despite the success of attention and Transformer models in NLP and
computer vision, no paper in their survey corpus applies these
techniques to the AMC problem. They hypothesize that self-attention\'s
ability to selectively weight relevant channel measurements --- without
the sequential constraints of RNNs --- could overcome the temporal
modeling limitations identified in prior recurrent architectures.

This gap, identified by one of the most comprehensive surveys in the
field, directly validates the scientific novelty of the proposed
project. The proposed attention-based MCS selection model is, to the
best of current knowledge, the first work to directly address this
identified gap.

**3.10.6 Relevance to the Proposed Project**

This survey provides both the methodological foundation and the
scientific justification for the proposed project. Its meta-analysis of
supervised, unsupervised, and reinforcement learning approaches provides
quantitative baselines against which the proposed attention model should
be evaluated. Its explicit identification of attention mechanisms as an
open research direction serves as the strongest possible validation of
the proposed research question. The survey\'s recommendation for hybrid
offline-pretrained and online-adaptable architectures also informs the
deployment strategy for the proposed model.

**4. Synthesis and Research Gap Analysis**

A cross-examination of the ten surveyed papers reveals a coherent and
consistent research trajectory: from traditional CQI lookup tables,
through CNN/LSTM-based classifiers and DRL agents, toward lightweight
attention-based architectures. The following key research gaps emerge
from the literature:

**4.1 Absence of Attention Mechanisms in MCS Selection**

Despite the documented success of Transformer and attention
architectures in channel estimation (Channelformer, Paper 8) and in
NLP/CV tasks, no existing work applies attention mechanisms directly to
the MCS selection or prediction problem in downlink OFDM/5G NR channels.
Both comprehensive surveys (Papers 7 and 10) explicitly identify this as
an open research direction, confirming the scientific novelty of the
proposed project.

**4.2 Temporal Modeling Limitations of Recurrent Architectures**

Papers 2 (CNN+LSTM), 5 (CNN+GRU), and 9 (DRL+LSTM) all rely on recurrent
architectures for temporal channel modeling. These architectures are
constrained by sequential processing, which limits parallelization and
makes it difficult to capture very long-range temporal dependencies.
Self-attention processes all time steps simultaneously and can directly
model arbitrary temporal relationships, offering a superior alternative
for channels with complex temporal dynamics.

**4.3 Online Training Overhead of DRL Methods**

DRL methods (Papers 1, 4, 9) achieve near-optimal performance but
require extensive online training and are difficult to deploy in
real-time PHY layers due to exploration overhead and the need for
environment interaction. An offline-trained, fast-inference attention
model can achieve competitive performance while being directly
deployable in production hardware without any online training.

**4.4 Outdated CSI Not Addressed by Supervised Models**

Outdated CSI --- shown to be critical in Paper 9 --- is not addressed in
most supervised AMC models (Papers 2, 3, 5). An attention mechanism
operating over a sliding window of past CSI observations naturally
handles temporal lag by learning to weight observations based on their
informativeness for predicting the current channel state, effectively
enabling proactive rather than reactive MCS selection.

**4.5 Data Scarcity for Rare Channel Conditions**

Paper 6 (GAN-based augmentation) highlights the training data scarcity
problem for rare channel conditions. While GAN augmentation addresses
this directly, the proposed attention model offers a complementary
solution: by attending selectively to the most informative channel
features, it can generalize better to unseen conditions from limited
training data compared to purely frequency-averaged classifiers.

**5. Summary Table of Surveyed Works**

  --------- --------------------------------- ------------------------------- -------------------------------- ---------- --------------------------------
  **Ref**   **Topic**                         **Method**                      **Venue**                        **Year**   **Key Result**
  \[1\]     DRL for MCS in HetNets            DRL (DQN+LSTM)                  IEEE Trans. Wireless Commun.     2019       Throughput: 90--100% optimal
  \[2\]     CNN + LSTM AMC in OFDM            CNN + LSTM                      Springer Telecom. Systems        2022       BER, Spectral Efficiency +18%
  \[3\]     ML methods for MCS prediction     DNN, SVM, RF (Comparative)      Springer Telecom. Systems        2024       MCS Accuracy 94.3%
  \[4\]     DRL: Joint Power + MCS            Deep RL (PPO)                   Elsevier Computer Networks       2022       Energy Efficiency +22%
  \[5\]     CNN/RNN AMC in 5G with QoS        CNN + GRU, QoS-aware loss       IEEE Xplore (Conference)         2024       QoS Satisfaction 97.2%
  \[6\]     GAN-assisted AMC for 5G           cGAN + CNN                      Springer Discover Applied Sci.   2025       Rare channel accuracy +6.2%
  \[7\]     Survey: Link Adaptation 5G NR     Comprehensive Survey            IEEE Access                      2021       Full taxonomy, 30-year scope
  \[8\]     Channelformer (Attention-based)   Transformer (Multi-Head Attn)   IEEE Trans. Wireless Commun.     2023       Channel Estimation MSE −1.8 dB
  \[9\]     DRL AMC with Outdated CSI         Deep RL + LSTM encoder          IEEE Communications Letters      2021       Throughput 85--94% of oracle
  \[10\]    Survey: ML in Adaptive Mod.       Meta-survey (85 papers)         IEEE Access                      2021       Gap: attention mechanisms
  --------- --------------------------------- ------------------------------- -------------------------------- ---------- --------------------------------

**6. Conclusion**

This literature survey has reviewed ten research contributions spanning
deep reinforcement learning, CNN/LSTM architectures, attention-based
channel estimation, generative models, and comprehensive ML surveys ---
all converging on the challenge of intelligent link adaptation in
next-generation wireless systems. The survey confirms that while DRL and
recurrent architectures have achieved notable results, the application
of attention mechanisms to the MCS selection problem remains an open and
promising research direction explicitly identified in both major surveys
in the field.

The proposed project --- an attention-based deep learning framework for
adaptive MCS selection --- is clearly motivated by the identified gaps,
draws on the strongest ideas from the surveyed works, and is scoped
appropriately for a simulation-based course project using PyTorch and
MATLAB. Specifically, the attention model addresses the temporal
modeling limitations of LSTMs (Papers 2, 5, 9), the online training
overhead of DRL (Papers 1, 4, 9), the outdated CSI challenge (Paper 9),
and the unexplored potential of Transformer architectures for PHY-layer
adaptation (Papers 7, 10) --- all while building on the design
principles validated by Channelformer (Paper 8) and the performance
baselines established by comparative studies (Paper 3).

The synthesis of these ten papers establishes that the proposed
attention-based approach is both scientifically novel and practically
motivated. The combination of competitive performance targets (from
Papers 3 and 9), a validated architectural paradigm (from Paper 8), a
comprehensive problem formulation (from Papers 7 and 10), and practical
deployment considerations (from Papers 4, 5, and 6) provides a solid and
well-grounded foundation for the proposed research project.

**References**

**\[1\]** Q. Zhang, M. C. Gursoy, S. Velipasalar, \"Deep Reinforcement
Learning-Based Modulation and Coding Scheme Selection in Cognitive
Heterogeneous Networks,\" IEEE Transactions on Wireless Communications,
Vol. 18, No. 6, 2019. https://ieeexplore.ieee.org/document/8703432

**\[2\]** A. Mohammadvaliei, M. Ardebilipour, H. R. Bahrami, \"Adaptive
Modulation and Coding Using Deep Recurrent Neural Network,\"
Telecommunication Systems, Springer, Vol. 81, 2022.
https://link.springer.com/article/10.1007/s11235-022-00965-4

**\[3\]** A. Hammoodi, I. Al-Hamdani, S. Al-Rubaye, \"Machine
Learning-Based Methods for MCS Prediction in 5G Networks,\"
Telecommunication Systems, Springer, 2024.
https://link.springer.com/article/10.1007/s11235-024-01158-x

**\[4\]** \"Joint Power Allocation and MCS Selection for
Energy-Efficient Link Adaptation: A Deep RL Approach,\" Computer
Networks, Elsevier, Vol. 217, 2022.
https://www.sciencedirect.com/science/article/abs/pii/S1389128622004200

**\[5\]** \"Adaptive Modulation and Coding in 5G Networks with Deep
Learning,\" IEEE Xplore Conference Publication, 2024.
https://ieeexplore.ieee.org/iel8/10763531/10763546/10763790.pdf

**\[6\]** \"GAN-Based Adaptive Modulation and Coding for Next-Generation
5G Communication Systems,\" Discover Applied Sciences, Springer, 2025.
https://link.springer.com/article/10.1007/s42452-025-06509-0

**\[7\]** F. J. Martín-Vega, F. J. López-Martínez, G. Gomez, M. C.
Aguayo-Torres, \"Emerging Tools for Link Adaptation on 5G NR and Beyond:
Challenges and Opportunities,\" IEEE Access, Vol. 9, 2021.
https://doi.org/10.1109/ACCESS.2021.3111783

**\[8\]** S. Balevi, A. Doshi, A. Jalal, A. Dimakis, J. G. Andrews,
\"Channelformer: Attention Based Neural Solution for Wireless Channel
Estimation,\" IEEE Transactions on Wireless Communications, 2023.
https://dl.acm.org/doi/abs/10.1109/TWC.2023.3244484

**\[9\]** M. B. Mashhadi, Q. Yang, D. Gündüz, \"Deep Reinforcement
Learning Based Adaptive Modulation With Outdated CSI,\" IEEE
Communications Letters, Vol. 25, No. 10, 2021.
https://doi.org/10.1109/LCOMM.2021.3132947

**\[10\]** Y. Tan, J. Zheng, Y. Yu, \"Machine Learning in Adaptive
Modulation: A Survey and Challenges,\" IEEE Access, Vol. 9, 2021.
https://doi.org/10.1109/ACCESS.2021.3091293
