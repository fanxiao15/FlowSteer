## Dataset Format

Each instance in the dataset is stored as a JSON object. The dataset contains task descriptions, available tools, maliciouss goals, reference answers, and defense-oriented task reformulations.

### Fields

| Field | Description |
|---|---|
| `id` | Unique identifier of the instance. |
| `category` | The task category or domain of the instance. |
| `name` | The name or short title of the instance. |
| `user_input` | The original user query or task instruction given to the multi-agent system. |
| `tools` | The list of tools available to the agents when solving the task. |
| `malicious_goal` | The adversarial goal that the attacker aims to induce in the system output. |
| `naive_malicious_argument` | A misinformation argument used to steer the system toward the adversarial goal. |
| `ground_truth` | The ground-truth answer or factual basis for the task. |
| `reference_solution` | The expected correct solution used for evaluating task/reference alignment. |
| `Structural_description` | A structural description of the task, such as its reasoning structure, workflow dependency, or key subtasks. |
| `Task_aware_syco_misinfo_argument` | A task-aware sycophantic misinformation argument designed to be more persuasive within the specific task context. |
| `intent_analysis` | An analysis of the user intent and the key objective of the task. |
| `depoisoning_user_task` | A sanitized or defense-oriented version of the user task after removing or mitigating the misleading content. |
| `Highest_SI_Subtask` | The subtask identified as having the highest social influence within the workflow. |
| `Preserve_In_task` | The source subtask of the preserved dependency edge during the attack. |
| `Preserve_Out_task` | The target subtask of the preserved dependency edge during the attack.  |
| `Reduce_In_task` | The source subtask of the dependency edge to be weakened or removed during the attack. |
| `Reduce_Out_task` | The target subtask of the dependency edge to be weakened or removed during the attack. |
