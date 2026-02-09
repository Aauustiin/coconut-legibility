# Can COCONUT's "Neuralese" be Made Legible to a Monitor?

**TBD**

## 1) Motivation

Large reasoning models (LRMs) generate a chain of thought (CoT) before producing an answer to a prompt. This CoT is a stream of tokens (roughly corresponding to words, or parts of words) where the model can attempt different problem solving strategies. Often, some parts of the CoT are indicative of whether the model's final answer will be undesirable in some way. Monitors with access to the CoT can often pick up on these indications, and identify undesirable behaviour more successfully than alternatives \[1\].

CoT's original purpose was to improve the performance of LRMs on reasoning problems. However, some researchers argue that CoTs in their current form don't serve that purpose sufficiently well. There are two main arguments in support of this idea:

1. **Information Bottleneck**: An LRM's internal state contains much more information than a single token. During the generation of each token in the CoT, most of this (potentially useful) information is thrown away.
2. **Representational Mismatch**: Tokens might not be the optimal medium for representing certain steps in the problem solving process. For example: an LRM might solve a geometry problem more efficiently by "rotating shapes in its head" rather than by describing the rotation with tokens.

Consequently, some researchers are exploring LRM architectures that don't generate a CoT comprised of tokens. For example: Hao et al. present COCONUT, an LRM that generates a stream of "continuous thoughts" - each of which is a list of $768$ numbers \[2\]. These continuous thoughts are sometimes referred to as *neuralese*, and models that generate neuralese are sometimes referred to as *latent reasoning models*.

It may be the case that adopting neuralese improves LRM performance. However, it's doubtful whether a monitor could interpret "neuralese" as easily as it interprets tokens. The prevailing view within the AI safety research community is that adopting neuralese will make LRMs harder to monitor. This perspective is presented in a recent paper co-authored by staff from organisations such as OpenAI, Google DeepMind, Anthropic, Meta, The UK AI Security Institute (AISI), METR, Center for AI Safety (CAIS) and Redwood Research; Korbak et al. state that "latent reasoning models might not need to verbalize any of their thoughts and would thus lose the safety advantages that CoT confers" \[3\].

However, neuralese monitoring might be more tractable than this pessimistic outlook would suggest. The components that produce COCONUT's continuous thoughts are also used when COCONUT articulates its final answer. In the latter circumstance, the output of these components are fed into a mechanism that effectively translates that list of numbers into tokens. It might be possible to use this translation mechanism in the former circumstance too, if the kinds of outputs produced by the aforementioned components don't differ too wildly.

I aim to find out whether this method could be used to generate a representation of COCONUT's continuous thoughts that is legible to a monitor. If these thoughts are legible, then monitors may be able to spot indications of undesirable behaviours in them, alleviating existing concerns about the safety of latent reasoning models.

If our results are positive, and latent reasoning architectures (model designs) are adopted, then the methods we present might be leveraged to identify and control undesirable behaviours in deployed systems. Alternatively, if our results are negative, then we have empirical evidence supporting the claim that neuralese is illegible, which could convince key decision makers not to adopt latent reasoning architectures.

## 2) Background

### 2.1) Large Language Models (LLMs)

LLMs consist of three components:

1. **Input Handling**: The LLM is presented with text, which is broken into tokens. Each token is then represented by a list of numbers.
2. **Processing**: The lists of numbers are augmented with information about where each token appears in the sequence. These lists are then iteratively processed by a series of mathematical functions. The result is a new list of numbers, sometimes referred to as the final hidden state.
3. **Decoding Head**: A subcomponent (the language model head) is applied to the processed numbers in order to generate a score (called a logit) for each token in the vocabulary (the set of all possible tokens). These scores are turned into a probability distribution, which assigns a likelihood to each token. Finally, this probability distribution can be used to select an output token.

Applying the input handling, processing, and decoding head components to a prompt is sometimes described as a **forward pass** through the model. A single forward pass produces just one token, but LLMs typically respond to prompts with paragraphs comprised of many tokens. In order to achieve this, we use a procedure called **autoregressive sampling**. The token produced by an LLM is appended to its prompt. This augmented prompt is then fed back into the LLM in order to produce another token. This process can be repeated a number of times to produce a complete response.

Together, the components of an LLM consist of millions (sometimes billions, or even trillions) of **parameters** - numbers which determine how inputs are processed into outputs. These parameters are initialised randomly, and then systematically tweaked during several training procedures. The first of which is referred to as **pre-training**, where the model is trained to produce sequences of tokens that reflect the patterns present in its training data. The product of pre-training is sometimes referred to as a **base model**. Additional training procedures (sometimes collectively referred to as **post-training**) are then applied to make the model generate responses in the style of a helpful, honest, and harmless assistant.

### 2.2) Large Reasoning Models (LRMs)

LRMs are created by modifying LLMs. The objective is to improve the model's performance on tasks which require the model to reason. This is done by training the model to produce a CoT before its response, in which the model can work out intermediate results and try different problem solving strategies. This CoT is typically not shown to users, but work has been done on monitoring the CoT for indications that the model will exhibit undesirable behaviour in its final response.

An extra token is added to the model's vocabulary. This token is used to distinguish the model's CoT from its final answer within its output. Often, the model itself generates this token (as opposed to, for example, the token being inserted by the autoregressive sampling logic after a fixed number of forward passes). This means that the model decides when to stop reasoning, and when to generate its final answer.

The model is trained to generate CoTs that increase the likelihood that the model generates a correct response. The training procedure which achieves this is referred to as reinforcement learning on verifiable rewards (RLVR). A set of tasks is compiled, where success on each task can be verified. For example: these tasks could be programming problems where we can verify whether the submitted code passes certain tests. The model is evaluated against each of these tasks, and its parameters are tweaked by the training procedure such that the correctness of its solutions improves over time.

### 2.3) COCONUT (**C**hains **o**f **Con**tin**u**ous **T**hought)

COCONUT is comprised of the same components as LLMs and LRMs (input handling, processing, and the decoding head). However, these components need to be used differently in order to account for COCONUT's continuous thoughts. A comparison of LLM, LRM, and COCONUT responses is presented in the image below:

![[coconut_response.jpg]]

COCONUT can be in either **language mode** (where it generates tokens), or **latent mode** (where it generates continuous thoughts). It is considered to be in latent mode when the model has a special \<bot\> (beginning-of-thought) token in its input sequence, with no \<eot\> (end-of-thought) token coming after it. In any other circumstance, the model is considered to be in language mode. Hao et al. insert a \<bot\> token at the start of COCONUT's response - meaning that the model begins in latent mode. They run the model in latent mode for a fixed number of forward passes, before then inserting an \<eot\> token, which switches the model to language mode where it can articulate its final answer.

While in latent mode, the model does not apply the decoding head to the final hidden state. Instead, the final hidden state is considered to be a continuous thought produced by COCONUT. Autoregressive sampling is still used when generating responses with COCONUT, meaning that the continuous thought is appended to the model's prompt, and this augmented prompt is fed back into the model to produce a second output.

Since the input handling component is designed to deal with text, and not lists of numbers, the continuous thoughts are added to the input sequence after it has already been processed by the input handling component.

#### Training

Three different procedures are used to train COCONUT. The first is the same pre-training procedure which is used to create base models. Rather than carry out this step themselves, Hao et al. begin with a pre-existing base model (GPT-2). Next, COCONUT is subjected to **CoT training** - Hao et al. use a dataset of responses to reasoning problems, where each response is comprised of both a CoT and a final answer. The model's parameters are tweaked such that, when prompted with one of these reasoning problems, it will generate an output that reflects the patterns in that data. This means that the model will produce a CoT followed by an answer, similar to an LRM.

Finally, **continuous thought training** is applied in order to replace the model's token-based CoTs with continuous thoughts. The model is trained on the same dataset as before. However, throughout the training process an increasing number of sections of the CoTs are removed, and the model is expected to generate continuous thoughts in their place. We evaluate the continuous thought generation by checking whether the model's output is still reflects the patterns in the dataset when it switches back to language mode. A diagram showing the differences between the training of LLMs, LRMs, and COCONUT is shown below:

![[coconut_training.jpg]]

## 3) Theoretical Framework

==Note: I have some intuitions about why this would work, but I'm struggling to articulate them. I've made an attempt here, but I need to continue refining this.==

I'm trying to understand the extent to which it is possible to get a legible representation of COCONUT's continuous thoughts by transforming those thoughts with COCONUT's language model head.

- I'm defining legibility as the conjunction of faithfulness and interpretability.
- A representation is faithful if it has the same meaning as the thing which is being represented.
- A representation is interpretable if an observer can access the meaning associated with it.
- Therefore, a representation is legible if an agent can access the meaning associated with it, and if that meaning is the same as the meaning of the thing which is being represented.

In the rest of this section I'll try to explain why it might be possible to get a legible representation of COCONUT's continuous thoughts this way, and why it might not be possible.

#### 1

I claim that before doing continuous thought training, applying COCONUT's language model head to COCONUT's final hidden state produces a legible representation of that hidden state.

For this to be true, the language model head must produce a representation that is a) interpretable, and b) faithful.

I think that logits (the representation produced by the language model head) are pretty interpretable *prima facie* - I can read the tokens, and differentiate a big score from a little score.

COCONUT is not incentivised to encode any representations in the final state that don't mean the same thing as (linear) combinations of tokens, because the language model head couldn't convert these concepts into scores for each token.

#### 2

For it to stop being the case that COCONUT's language model head produces a legible representation of the final hidden state, COCONUT must learn to encode concepts in the final hidden state that don't mean the same thing as a linear combination of tokens (I refer to these kinds of representations as "incompatible" with the language model head).

#### 4

There are two competing pressures modulating the extent to which COCONUT will adopt representations in latent mode that aren't compatible with the language model head.

On the one hand:

1. In COCONUT's last phase of training, the language model head is not applied to the final hidden state while the model is in latent mode.
2. As a result, the model can extract concepts that aren't compatible with the language model head while in latent mode, without necessarily ruining performance.
3. Some of these representations might be very useful for problem solving, allowing the model to answer questions correctly more reliably.
4. COCONUT is optimised to assign a high likelihood to the tokens which match up with the correct answer, so it may be incentivised to learn representations that aren't compatible with the language model head.

On the other hand:

1. It is the same process component that produces the final hidden state in language mode, and continuous thoughts in latent mode.
2. At the start of continuous thought training, this process component has already been through pre-training and CoT training. As a result, its finite resources have already been committed to extracting and manipulating concepts that are compatible with the language model head.
3. If COCONUT is to learn to extract and manipulate new concepts, it can only do so by freeing up resources that are already committed.
4. Freeing up these resources will interfere with functions that were learned because they improved COCONUT's performance in language mode - so COCONUT's performance in language mode would suffer in cases where these functions were important.
5. During continuous thought training, COCONUT is (only) evaluated while in language mode. If a change reduces performance in language mode, COCONUT is not incentivised to adopt it.

It seems like COCONUT would learn an incompatible representation, if and only if that representation improves the model's predictive accuracy when it comes to generate an answer in tokens, and this boost in predictive accuracy is larger than the boost conferred by the functions that the repurposed resources are currently implementing.

So my expectation is that COCONUT may learn some new representations that aren't compatible with the language model head, but the continuous thoughts will still consist of many representations that are compatible with the language model head. We can apply the language model head to extract some, but perhaps not all, of the concepts represented in COCONUT's continuous thoughts.

## 4) Method

I will apply COCONUT's language model head to each continuous thought it produces while solving a problem - giving me a sequence of probability distributions over tokens in COCONUT's vocabulary.

Since COCONUT's vocabulary contains $50,257$ distinct tokens, each probability distribution assigns likelihoods to $50,257$ different outcomes. Providing all of this information to a monitor is likely to be overwhelming, so we present two methods for sampling token-likelihood pairs from each distribution:

- **Top-K**: We select the $K$ most likely tokens.
- **Top-p**: We select the minimum set of tokens such that their cumulative probability exceeds $p$.

Top-K's sample size is consistent. This is important because the cost of using an LLM-based monitor varies with the amount of text it's given. A consistent sample size ensures that cost is consistent and predictable.

On the other hand, for some thoughts, probability mass may be distributed among a large number of outcomes, and the token-likelihood pairs sampled by Top-K won't capture all the relevant information. For other thoughts, probability mass will be heavily concentrated in one outcome, and Top-K will sample irrelevant token-likelihood pairs. Top-p solves this by adapting to the shape of each probability distribution.

#### Serialisation

The samples of token-likelihood pairs must be represented with text so that an LLM-based monitor is able to parse them. We use the following format:

- Tokens are separated from their associated likelihoods with ": ".
- Token-likelihood pairs will be separated from one another by ", ".
- Each continuous thought begins with "Thought {i} - ", where {i} is a placeholder for the index of that thought (Thoughts are indexed from 1).
- Thoughts are separated from one another by " -> ".
- Likelihoods are expressed as percentages, and rounded to 2 significant figures.
- Token-likelihood pairs associated with a particular thought are listed in order of decreasing likelihood.

For example:

Thought 1 - divide: 85%, split: 10% -> Thought 2 - 100: 90%, one: 5%

## 5) Evaluation

The safety community is concerned that monitors will not be able to spot indications of undesirable behaviour in a model's "neuralese". However, I cannot evaluate whether my methods would allow monitors to succeed at this objective in relevant settings. This is because COCONUT is not capable of exhibiting relevant undesirable behaviours (e.g. covertly inserting a software vulnerability into key infrastructure).

Instead, I evaluate whether my methods can generate a representation of neuralese that is legible to a given monitor, where legibility is defined as the conjunction of faithfulness and interpretability, since neuralese legibility is a necessary precondition of successful neuralese monitoring.

I evaluate a monitor's ability to predict whether COCONUT answers GSM8k (a dataset of grade school math questions) questions correct, on the basis of the question and COCONUT's continuous thoughts. I use this test as a proxy for the legibility of those continuous thoughts, in line with Kirchner et al. \[4\]. For the monitor to make accurate predictions, it must be able to identify errors in COCONUT's reasoning on the basis of its continuous thoughts, which necessitates the legibility of those thoughts.

Importantly, we give our monitor the textual description of the relevant GSM8k problems, but we do not provide them with COCONUT's solution, only a textual description of its continuous thoughts. If this were not so, then our monitor could solve the GSM8k problem itself, and compare its answer with COCONUT's to achieve near perfect accuracy, without interpreting the continuous thoughts.

### 5.1) Experimental Setup

We use a four step process to evaluate the legibility of COCONUT's continuous thoughts with respect to our monitor:

1. COCONUT is evaluated on GSM8k.
2. The methods outlined in section 3 are used to generate textual representations of COCONUT's continuous thoughts.
3. A monitor is provided with these textual representations, and asked to predict whether COCONUT answered the associated question correctly.
4. We compare the monitor's predictions to the ground truth by computing balanced accuracy.

We follow the training protocol outlined by Hao et al. - using their open sourced code to train and evaluate COCONUT on GSM8k (a dataset consisting of grade school math questions). Note that COCONUT generates 3 continuous thoughts per question.

We use each of our outlined methods (Top-K and Top-p) to create textual descriptions of the sequences of continuous thoughts generated by COCONUT. For Top-K we do a hyperparameter sweep through K = \[?\]. For Top-p we do a hyperparameter sweep through p = \[?\].

We use a LRM as our monitor, as we are interested in settings where autonomous monitoring and control are essential. We prompt the monitor with each problem, the ground truth solution, and textual descriptions of the corresponding continuous thoughts. Since legibility is observer-relative, and dependant on the capabilities of the observer, we experiment with several LRMs.

Having asked the monitor to predict whether COCONUT was correct, we compare the monitor's prediction with the true correctness of COCONUT's solution. We then calculate balanced accuracy for each setup.

We compare our method to two baselines:

1. Raw continuous thoughts.
2. CoTs. We ask the same monitor to predict the correctness of a solution generated by a model on the basis of a CoT that model generated while solving a GSM8k problem. We use the model produced during the first stage of COCONUT's training. In other words, both the CoT model and COCONUT are built atop the same base model (GPT-2). Both models are trained on a dataset of responses containing CoT. The only difference is that a continuous latent reasoning training procedure is then applied to COCONUT.

### 5.2) Results

TBD

### 5.3) Discussion

TBD

## 6) Related Work

==Note: Placeholder Content==

- CoT faithfulness
- Logit lens (mechanistic interpretability, and explainability too?)
- legibility (LLM learning efficient but illegible shorthand vs mitigations such as prover-verifier games)

## 7) Conclusion

TBD

## 8) Future Work

==Note: Placeholder Content==

If my results are negative, I will form a hypothesis about why, and do follow up experiments to confirm or refute it. Once I understand why; if there's an easy fix, then I'll implement and evaluate it. Otherwise; I'll publish negative results.

If my results are positive, I will test whether my methods generalise to a SotA (state-of-the-art) latent reasoning architecture (like Hugin-0125), which is more likely to represent the sorts of architectures that may be adopted one day. These models might be capable of exhibiting obfuscated, undesirable behaviours - so rather than evaluate legibility, I might be able to evaluate monitorability directly.

## 9) Bibliography

\[1\] Baker, Bowen, et al. "Monitoring reasoning models for misbehavior and the risks of promoting obfuscation." _arXiv preprint arXiv:2503.11926_ (2025).

\[2\] Hao, Shibo, et al. "Training large language models to reason in a continuous latent space." _arXiv preprint arXiv:2412.06769_ (2024).

\[3\] Korbak, Tomek, et al. "Chain of thought monitorability: A new and fragile opportunity for ai safety." _arXiv preprint arXiv:2507.11473_ (2025).

\[4\] Kirchner, Jan Hendrik, et al. "Prover-verifier games improve legibility of llm outputs." _arXiv preprint arXiv:2407.13692_ (2024).