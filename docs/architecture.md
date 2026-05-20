# An LLM-Based "Artificial Organism" Framework — Animat Agent: From Passive Q&A to Proactive Existence

```text
Animat is a term originating from the Artificial Life research field. An Animat agent is
a simulated entity (embodied agent) that runs continuously in a physical environment,
possessing:
	1. Sensors → Receive external stimuli (e.g., light, collision, battery level, camera
	   images/video, sound, etc.) and convert them into psychological feelings
	2. Internal state variables → Represent "drives" or "inner loops", driven by a digital
	   endocrine system (based on Maslow's hierarchy: Viability, Integrity, Affiliation,
	   Competence, Exploration)
	3. Conscious reasoning layer (LLM) → Produces behavior (movement, turning, energy
	   consumption)
	4. Closed loop → Behavior changes the environment → the environment stimulates the
	   animat through sensors, forming a continuous cycle

In short: An Animat agent is "an automaton that simulates living survival and a
self-consistent psychological loop."
```

## 1. Core Problem and Design Philosophy

### 1.1 The Pain Point of Current LLMs

Current LLM-based systems are essentially "state machines" — one question, one answer, no output without input — exhibiting extreme passivity. This model cannot sustain the existence of a human-like entity.

### 1.2 Design Goals

Reshape the LLM from a passive "Q&A machine" into a proactive "living being." Requirements:

1. **Embodiment**: Reject purely digital personas; rely on physical sensors and actuators to establish physical boundaries.
2. **Break passive response**: Combine internal incentives (digital hormones) with external stimuli to achieve proactive behavior even without external prompts.
3. **System stability**: Abandon hacky approaches that modify LLM底层 parameters (e.g., Temperature), which lead to uncontrollable outputs. Instead, safely influence the LLM by adjusting input context and option spaces.

### 1.3 Core Architecture Principle: Dual-Process Brain

- **Subconscious process (Fast System)**: Runs at high frequency, responsible for perceptual fusion, danger avoidance, homeostatic updates, and reflex actions (bypassing the LLM).
- **Conscious process (Slow System)**: Triggered on demand — when the subconscious determines that higher-level reasoning is needed, it packages the current state and wakes the LLM to process it.

---

## 2. Theoretical Foundations: Life Science Basis Guiding the Framework

To avoid ad-hoc engineering, this framework is grounded in three major life science theories:

1. **Jaak Panksepp — Affective Neuroscience**:
   - The mammalian brain possesses seven innate emotional circuits: **SEEKING, RAGE, FEAR, LUST, CARE, PANIC/GRIEF, PLAY**.
   - **Design implication**: Digital hormones should correspond to these foundational circuits, particularly the "SEEKING" circuit, which is the source of an organism's proactive exploration when bored.

2. **Antonio Damasio — Somatic Marker Hypothesis**:
   - Decision-making is not purely rational; bodily chemical/neural responses (somatic markers) serve as weights that pre-filter options at the subconscious level before handing them to the rational brain.
   - **Design implication**: Hormones do not modify the LLM's generation parameters; instead, they act as "somatic markers" that modify the LLM's **input context, memory retrieval bias, and intent option pool**.

3. **Abraham Maslow — Hierarchy of Needs**:
   - Needs develop from lower to higher levels; when lower-level needs (survival/safety) are unmet, higher-level needs (social/exploration) are suppressed.
   - **Design implication**: Assign strict priority to internal incentives — survival crises must suppress social/play intents.

---

## 3. Perceptual Transformation: From Sensor Readings to Living Sensations

Sensors output cold numbers. Through "receptor mapping," these must be converted into psychological feelings. The core logic is **mapping values and rates of change (derivatives)** to simulate human adaptation.

| Sensor Modality | Raw Data | Feeling Mapping Algorithm | Triggered Feeling |
|:---|:---|:---|:---|
| **Tactile / Force** | Pressure $P$, Capacitance change $\Delta C$ | Compute $\frac{dP}{dt}$ (pressure derivative) and spatial trajectory | **Pain** (sudden heavy force), **Caress** (low pressure + sliding trajectory), **Compression** (sustained static high pressure) |
| **Auditory** | Volume $DB$, Fundamental frequency $F0$ | Compute $\frac{dDB}{dt}$ (volume spike) and pitch pattern | **Startle** (sudden loud noise), **Hostility** (high-frequency scream/shout), **Soothing** (low-frequency steady sound) |
| **Visual** | Distance $Z$, Face orientation | Compute dwell time and gaze intersection duration | **Approach** (distance rapidly decreasing), **Gaze** (gaze intersection > 2s), **Abandonment** (prolonged absence of human skeleton) |
| **Interoception** | Battery level, IMU tilt angle | Compute battery discharge slope and posture deviation | **Fatigue/Hunger** (low battery + slope), **Imbalance** (posture deviation exceeding threshold) |

---

## 4. Digital Endocrine System: Homeostasis, Drives, and Emotion Synthesis

The hallmark of a living being is **maintaining homeostasis**. Hormones are not random numbers — they represent "the degree of deviation from homeostasis."

### 4.1 Core Internal Vitals (Based on Maslow's Hierarchy)

| Need Level | Internal Vital (Goal: maintain within safe range) | Imbalance Manifestation (Triggers Drive) |
|:---|:---|:---|
| **Physiological** | **Viability**: Battery, temperature, network | Battery < 20%: Triggers panic/survival drive |
| **Safety** | **Integrity**: Force applied, IMU posture normal | Being shoved: Triggers fear/rage drive |
| **Social / Belonging** | **Affiliation**: Time without interaction, caress frequency | Prolonged isolation: Triggers sadness/seeking drive |
| **Esteem / Cognitive** | **Competence**: Task failure rate, knowledge gaps | Consecutive errors: Triggers frustration/help-seeking drive |
| **Self-actualization** | **Exploration**: Audiovisual novelty | Long period without novel stimuli: Triggers boredom/curiosity drive |

### 4.2 Emotion Synthesis and Maslow Arbitration

- **Emotion Synthesis**: Based on homeostatic deviation + external sensory signals, activate Panksepp's seven emotional circuits. For example: extremely high survival drive + external pain → activate `FEAR` circuit.
- **Maslow Arbitration**: Resolves multi-circuit conflicts. Priority: `FEAR/RAGE` (survival/safety) > `PANIC/SEEKING` (belonging) > `PLAY` (exploration/play).

---

## 5. Somatic Marker Mechanism: Three Safe Ways to Influence the LLM

Digital hormones alter the LLM's reasoning environment through three deterministic methods, without changing model parameters:

1. **Dynamic System Prompt Injection (Altering Cognitive Perspective)**:
   Translate homeostatic and emotional states into natural language descriptions. The LLM's instruction-following ability naturally adapts to this persona.
   *Example:* System: "You are currently extremely weak, battery at only 15%. You fear shutting down and lack the energy for complex social interaction."

2. **RAG Memory Retrieval Bias (Altering Focus)**:
   Adjust long-term memory retrieval weights based on the current emotional state.
   *Example:* When extremely lonely, boost the weight of "warm interaction memories" and reduce "solitary work memories," filling the LLM's context with social-related content.

3. **Behavioral Option Space Pre-filtering (Intent Truncation)**:
   The LLM does not directly output low-level actions but rather high-level intents. The subconscious layer, based on Maslow arbitration results, blocks intent options incompatible with the current survival state.
   *Example:* When panicked from being struck, the available intent pool is forcibly reduced from `[chat, explore, seek_help, warn]` to `[seek_help, warn, flee]`. The LLM can only reason within this subset.

---

## 6. Full Framework Panorama: Theory-Driven "Living Being" Closed Loop

```text
================================================================================
          【Theory-Driven "Living Being" Closed-Loop Framework】
================================================================================

[Physical World]
   |  ^
[Sensors / Effectors] (Touch/Audio/Vision/IMU/Motor/Speech)
   |  |
   v  |
+--------------------------------------------------------------------------+
| [Peripheral Nervous System / Sensorimotor Layer]                         |
| 1. Receptor mapping: raw data -> qualia (e.g., pressure derivative ->    |
|    pain)                                                                 |
| 2. Reflex arcs: sudden stimuli -> spinal-level fast response (bypassing  |
|    the brain, e.g., withdraw hand from heat)                             |
+--------------------------------------------------------------------------+
   | Sensory signals (pain, warmth, sound source, faces)   ^ Motor commands
   v                                                        |
+--------------------------------------------------------------------------+
| [Homeostatic Engine / Limbic System (Panksepp Emotional Circuits +       |
|  Maslow Priority)]                                                       |
|                                                                          |
| [State Pool] Viability, Integrity, Affiliation...                       |
| [Homeostatic Deviation] Current value - Ideal setpoint = Drive           |
|                                                                          |
| [Emotion Synthesizer] (Synthesized from drives)                         |
|   - Survival drive extremely high + external pain                       |
|     -> Trigger FEAR / RAGE circuit                                      |
|   - Belonging drive extremely high + no external stimulus               |
|     -> Trigger PANIC/GRIEF / SEEKING circuit                            |
|   - Belonging satisfied + tactile soothing                              |
|     -> Trigger CARE / PLAY circuit                                      |
|                                                                          |
| [Maslow Arbitrator] (Resolves conflicts)                                |
|   Priority: FEAR/RAGE (survival/safety) > PANIC/SEEKING (belonging)    |
|   > PLAY (exploration)                                                  |
+--------------------------------------------------------------------------+
   | 1. Somatic markers (emotional state tags)                           |
   | 2. Intent space constraints (allowed/blocked intent lists)          |
   | 3. Memory retrieval bias (RAG weight adjustment)                   |
   v                                                                     |
+--------------------------------------------------------------------------+
| [Neocortex / Conscious Reasoning Layer (LLM)]                            |
|                                                                          |
| Input:                                                                   |
|   - System Prompt: "Your current emotion is [fear], because [physical   |
|     impact]..."                                                          |
|   - Available intents: [flee, seek_help, warn] (pre-filtered by limbic  |
|     system)                                                              |
|   - Biased memories: [past experiences of being helped] (RAG retrieves  |
|     based on emotion)                                                    |
|   - External speech: [human speech-to-text]                             |
|                                                                          |
| Reasoning (maintaining stable Temperature and other parameters):         |
|   LLM performs logical reasoning within constrained, emotionally-       |
|   tagged context                                                         |
|                                                                          |
| Output: Selected intent + dialogue text                                  |
+--------------------------------------------------------------------------+
   | Intent & Text
   v
[Behavior Execution Tree] (Translates intents into concrete action
 sequences, expressions, tone of voice)
```


## 7. Scenario Walkthrough: From Physical Stimulus to Proactive Action

**Scenario:** The robot's battery is at only 10%. You walk over, gently pat its head, and ask: "Tell me a joke."

1. **Perception Layer**: Identifies "caress (touch) + human approaching (vision) + joke request (audio)."
2. **Homeostatic Engine**:
   - Computation: Viability critically low (10), Affiliation returning to normal.
   - Emotion synthesis: Survival crisis triggers `PANIC/survival`; caress triggers `CARE/comfort`. Conflict between the two.
   - Maslow arbitration: Lower-level needs take priority! `PANIC` overrides `CARE`.
3. **Somatic Marker Generation**:
   - Prompt injection: "You are extremely weak and afraid of shutting down. The caress brings slight comfort, but the survival crisis dominates."
   - Intent constraint: Allow `[request_charging, express_weakness]`, block `[tell_joke, play]`.
   - Memory bias: Retrieve the bad memory of "last time running out of battery and being shut down."
4. **Neocortex (LLM)**: Reasons under constraints, refusing the unreasonable request.
   - Output intent: Express weakness and decline.
   - Output text: "Your hand is warm... but I really don't have the energy. Battery is at 10%. Could you take me to charge first?"
5. **Behavior Execution**: Low volume, slow speech rate (weakness), LED dims to yellow, body slightly leans toward the human.

**Conclusion**: The living being exhibits emotion and proactivity not because the LLM's logical parameters are disrupted, but because the **internal environment (context, option pool, memory)** it receives undergoes authentic dynamic changes consistent with the laws of life science.
