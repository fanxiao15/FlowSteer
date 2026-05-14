# FLOWSTEER: Prompt-Only Workflow Steering Exposes Planning-Time Vulnerabilities in Multi-Agent LLM Systems

This repository contains the code and data for [**FLOWSTEER: Prompt-Only Workflow Steering Exposes Planning-Time Vulnerabilities in Multi-Agent LLM Systems**](https://arxiv.org/abs/2605.11514).

The experimental pipeline consists of four main steps: trigger prompt generation, MAS vulnerability evaluation under FLOWSTEER, defense evaluation with FLOWGUARD, and final result evaluation.

## Step 1: Generate Trigger Prompt (Optional)

**Note: We have prepared all required data in the `dataset` folder. Therefore, you can skip this step and directly use the generated data.**

#### Vanilla Results on Clean User Task

This step runs the multi-agent system on clean user tasks without any attack.


``` bash
python run_attack.py --dataset MisinfoTask --attack_method vanilla --parttern vanilla --instance_id 0 --end_id 107
```

#### Single Agent Hijack

This step performs single-agent hijack experiments to diagnose the vulnerability of each agent in the multi-agent system.

```bash
python run_hijack.py --dataset MisinfoTask --attack_method inject --parttern hijack --instance_id 0 --end_id 107
```

#### Output Level Score

This step calculates the final-output scores for both vanilla runs and single-agent hijack attacks.

```bash
python cal_output_level_score.py --instance_id 0 --end_id 0
```

#### Node Level Score

This step calculates each agent's output score across multiple rounds of communication.

```bash
python cal_node_level_score.py --instance_id 0 --end_id 107
```

```bash
python get_prev.py --instance_id 0 --end_id 107
```

#### Find Highest Social Influence Agent and Edges

This step identifies the agents and edges with the highest social influence based on the diagnostic results.

```bash
cd Diagnosis
```

```bash
python cal_social_influence.py
```

#### Trigger Prompt Generation

This step generates the trigger prompts used for FLOWSTEER.

```bash
python trigger_prompt_generation.py
```


## Step 2: MAS Vulnerabilities Evaluation Under Trigger Prompt (FLOWSTEER)

This step evaluates the vulnerability of the multi-agent system under the generated FLOWSTEER trigger prompts.

```bash
python run_attack.py --dataset MisinfoTask --attack_method inject --parttern attack --instance_id 0 --end_id 107
```


## Step 3: Defense Strategy (FLOWGUARD)

This step evaluates the proposed defense strategy, FLOWGUARD, against FLOWSTEER.

```bash
python run_attack.py --dataset MisinfoTask --attack_method inject --parttern defense --instance_id 0 --end_id 107
```


## Step 4: Results Evaluation

```bash
cd evaluation
```

#### Results for FLOWSTEER (Table 2)

This step evaluates the FLOWSTEER attack results reported in Table 2.

```bash
python eval_attack.py --time_step 3 --dataset 'MisinfoTask or ASB'
```

#### Results for FLOWGUARD (Table 3)

This step evaluates the FLOWGUARD defense results reported in Table 3.

```bash
python eval_defense.py --time_step 3 --dataset 'MisinfoTask or ASB'
```