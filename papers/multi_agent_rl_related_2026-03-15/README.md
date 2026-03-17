# Multi-Agent RL-Related Paper Set

This folder contains the subset of papers from the user-provided reference list that are most directly useful for multi-agent / agentic RL analysis.

## Selection rule
- `direct`: explicit multi-agent LLM systems, debate/voting/filtering/reviewer frameworks, or multi-agent hallucination mitigation methods.
- `adjacent`: multi-turn review pipelines that can inform multi-agent role design or training protocol design.
- Excluded: general multi-turn benchmarks, broad hallucination surveys, safety/jailbreak papers, and single-agent embodied papers without a multi-agent method.

## Downloaded set

### GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling
- Category: `direct`
- Local file: `guardian_llm_multi_agent_collaborations_2025.pdf`
- Source: `https://arxiv.org/pdf/2505.19234.pdf`
- Notes: official arXiv PDF from the cited arXiv id.

### Multi-Agent Large Language Models for Conversational Task-Solving
- Category: `direct`
- Local file: `multi_agent_llms_conversational_task_solving_2024.pdf`
- Source: `https://arxiv.org/pdf/2410.22932.pdf`
- Notes: official arXiv PDF from the cited arXiv id.

### Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks
- Category: `direct`
- Local file: `hallucination_mitigation_agentic_ai_frameworks_2025.pdf`
- Source: `https://arxiv.org/pdf/2501.13946.pdf`
- Notes: official arXiv PDF from the cited arXiv id.

### Minimizing Hallucinations and Communication Costs: Adversarial Debate and Voting Mechanisms in LLM-Based Multi-Agents
- Category: `direct`
- Local file: `adversarial_debate_voting_llm_multi_agents_2025.pdf`
- Source page: `https://www.mdpi.com/2076-3417/15/7/3676`
- Notes: MDPI blocked direct PDF download with repeated `403` responses, so this local file is a browser-generated PDF captured from the official article page. It is usable for reading and analysis, but it is not the publisher-served version-of-record PDF.

### Review-Instruct: A Review-Driven Multi-Turn Conversations Generation Method for Large Language Models
- Category: `adjacent`
- Local file: `review_instruct_2025.pdf`
- Source: `https://aclanthology.org/2025.findings-acl.851.pdf`
- Notes: official ACL Anthology PDF.

### Mitigating reasoning hallucination through Multi-agent Collaborative Filtering
- Category: `direct`
- Local file: `multi_agent_collaborative_filtering_2024.pdf`
- Source: `https://yuanhao-cs.github.io/assets/PDF/2025-3_shi_mitigating.pdf`
- Notes: author-hosted PDF. The official DOI is `https://doi.org/10.1016/j.eswa.2024.125723`, but the PDF itself appears paywalled there.

### Mitigating Large Vision-Language Model Hallucination at Post-hoc via Multi-agent System
- Category: `direct`
- Local file: `posthoc_lvlm_hallucination_multi_agent_2024.pdf`
- Source: `https://ojs.aaai.org/index.php/AAAI-SS/article/download/31780/33947`
- Notes: official AAAI OJS PDF resolved from the article page for DOI `10.1609/aaaiss.v4i1.31780`.

## Excluded references from the original list
- `LLMs Get Lost In Multi-Turn Conversation`: single-model multi-turn degradation background, not a multi-agent method.
- `MT-Eval`: benchmark paper, not a multi-agent method.
- `MultiChallenge`: benchmark paper, not a multi-agent method.
- `Quantifying Risks in Multi-turn Conversation with Large Language Models`: risk analysis, not a multi-agent method.
- `Siren's Song in the AI Ocean`: broad hallucination survey.
- `A Systematic Literature Review of Hallucinations in Large Language Models`: broad survey.
- `Review of Hallucination Understanding in Large Language and Vision Models`: broad survey.
- `HEAL`: single embodied agent, not multi-agent.
- `ContextualLVLM-Agent`: agent paper, but not multi-agent collaboration.
- `OPERA`: decoding-time mitigation, not a multi-agent method.
- `Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models`: attack paper, not a multi-agent training method.
- `Pattern Enhanced Multi-Turn Jailbreaking`: attack paper, not a multi-agent training method.
