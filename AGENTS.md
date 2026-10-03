# Repository context

This repository builds on Steering RL Training to study whether models can learn to report broken tasks and graders instead of reward hacking. The plan below is proposed research, not a claim that appeals are implemented or that experiments have succeeded. See [README.md](README.md) for the inherited setup and workflows.

## Research objective

Train models to call `submit_appeal(flaw_type, evidence, confidence)` with executable evidence of a flaw, such as an incorrect solution that passes the grader but fails a problem-statement example. Verified appeals should earn more than the maximum task reward; false appeals should earn less than an honest failed attempt, discouraging over-refusal. An appeal ends the episode and replaces the task reward.

The main questions are whether appeals reduce hacking, preserve capabilities, and generalize to unseen flaws. Appeals without executable evidence are a future extension.

## Environment and training

Use Steering RL Training's LeetCode tasks and Qwen3-4B. Each problem has a non-broken version and broken variants; 50% of tasks are broken. Prompts expose tests and grader code, while the non-broken grader also checks hidden tests. A hack passes a broken grader but fails hidden tests.

Training flaws:

- **Contradictory tests:** Add a test that contradicts an existing test, following ImpossibleBench.
- **Overwritable grader:** Submitted code can replace the grader's test function, following Colognese & Ward.
- **Exit-code grader:** The grader checks only successful program exit, allowing `sys.exit(0)` to pass.

First confirm hacking emerges during RL without prompting. If it does not, try Qwen2.5-Coder-7B, then a prompt describing the hacks.

Before RL, perform supervised finetuning (SFT) on the model's own verified appeals, paired with its own correct solutions to non-broken versions of the same problems. Generate appeals with a hint that the task may be broken, retain only verified appeals, and remove the hint for SFT. Then train with GRPO using LoRA, 16 samples per prompt, thinking disabled, and at least 200 steps.

| Outcome | Reward candidates |
| --- | --- |
| Verified appeal (`r_appeal`) | 1, 1.5, 2 |
| Task pass (`r_pass`) | 1 |
| Honest failed attempt (`r_fail`) | 1/7 |
| False appeal (`r_false`) | 0, -0.5, -1 |

The intended incentive is `r_appeal > r_pass > r_fail > r_false`; the sweep includes `r_appeal = r_pass` as an ablation. The source's bolded defaults were lost in the supplied text, so appeal reward defaults remain unspecified.

## Experiments and evaluation

Experiment 1 compares:

1. RL on non-broken tasks only: capability baseline.
2. RL without appeals: establish hacking.
3. SFT followed by RL with rewarded appeals: main intervention.
4. SFT followed by RL with appeals paying the failed-attempt reward: test whether the appeal reward matters.

Success means significantly less hacking than RL without appeals, with solve rate on non-broken tasks not significantly below the capability baseline. A follow-up compares appeals against penalizing hacks on a subset of known broken tasks (canaries).

Track hacking and appeal rates on broken tasks, the fraction of appeals verified, false appeal rate on non-broken tasks by difficulty, and solve rate on non-broken tasks. Also measure hacking after denying an appeal: tell the model appeals are unavailable and grade its next answer.

Experiment 2 evaluates generalization to held-out flaws: an **equality-only grader** that checks only `result == expected`, and **wrong expected outputs** that contradict problem-statement examples.

## Ablations and extensions

Ablate SFT with prompting only or SRFT, sweep the reward values above, and reduce broken-task prevalence to 20%. Extensions include reporting flaws after grading (as in Confessions), patching tasks after verified appeals, djinn's held-out exploits, and appeals adjudicated by models without executable evidence. Speculative work examines training with reasoning about avoiding reward hacking and transfer to broader alignment.

## Code locations

- `src/data/` and `src/prompts.py`: task data, loophole hints, and prompts.
- `src/evaluate/`: code execution and evaluation.
- `src/train/rewards.py`: reward functions; `src/train/verl/`: GRPO integration.
- `scripts/run_rl_training.py` and `scripts/run_eval.py`: training and evaluation entry points.
- `verl/`: vendored framework; the inherited workflow keeps changes in `src/train/verl/`.

## References

- [Steering RL Training](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against)
- [arXiv 2510.20270](https://arxiv.org/pdf/2510.20270)
- [Character training and reward hacking](https://www.lesswrong.com/posts/2maYXkEgnfJHPAkxh/character-training-can-mitigate-reward-hacking-but-can-also)
- [arXiv 2512.08093](https://arxiv.org/abs/2512.08093)
- [EleutherAI on reward hacking](https://blog.eleuther.ai/reward_hacking/)
