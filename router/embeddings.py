from __future__ import annotations

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from .models import EmbeddingScore, Stage


class EmbeddingRouter:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.stage_exemplars: Dict[Stage, List[str]] = {
            Stage.EXPLORATION: [
                "map the space of possible approaches",
                "what are the structurally distinct framings here",
                "brainstorm alternatives before committing",
                "list several different ways to define the problem",
                "generate multiple interpretations before choosing one",
                "open up the option space instead of converging early",
                "explore competing hypotheses for why this happened",
                "identify radically different solution paths",
                "surface hidden assumptions by reframing the question",
                "propose 5 diverse directions to investigate",
                "compare fundamentally different strategies",
                "find new angles we have not considered",
            ],
            Stage.ADVERSARIAL: [
                "stress test this interpretation",
                "what would break this frame",
                "find the strongest objection to this plan",
                "enumerate failure modes before launch",
                "attack the weak points in this argument",
                "where is this proposal brittle under pressure",
                "identify the top destabilizing risk",
                "challenge this approach with worst case scenarios",
                "red team this strategy and expose vulnerabilities",
                "what assumptions make this likely to fail",
                "find contradictions that could collapse the plan",
                "simulate how this could go wrong in production",
            ],
            Stage.SYNTHESIS: [
                "integrate these scattered signals into one coherent frame",
                "compress many observations into a clear organizing logic",
                "what is the strongest interpretation of all this",
                "connect the parts into a single explanatory spine",
                "summarize the core pattern tying these cases together",
                "derive one unifying model from competing evidence",
                "distill the narrative thread across these fragments",
                "turn this messy analysis into a concise structure",
                "identify the central mechanism behind these outcomes",
                "produce the simplest frame that explains most signals",
                "synthesize findings into a shared mental model",
                "resolve conflicting inputs into a coherent conclusion",
            ],
            Stage.EPISTEMIC: [
                "what evidence supports this claim",
                "separate known facts from assumptions",
                "identify what we still do not know",
                "verify whether this conclusion is actually justified",
                "assess confidence levels for each assertion",
                "what data would falsify this hypothesis",
                "audit the quality of evidence behind the argument",
                "mark unsupported statements and needed proof",
                "list uncertainty sources that block confidence",
                "distinguish speculation from validated findings",
                "check if our reasoning overreaches the data",
                "define what would count as decisive evidence",
            ],
            Stage.OPERATOR: [
                "produce a concrete decision with next steps",
                "choose one option and justify the tradeoff",
                "what should we do this week",
                "commit to a plan and define immediate actions",
                "make the call now under time pressure",
                "select the best path and assign owners",
                "decide between these two alternatives today",
                "recommend one course of action and fallback",
                "turn analysis into an executable action plan",
                "prioritize tasks for the next sprint",
                "ship a decision memo with clear commitments",
                "define the next move and trigger for revision",
            ],
            Stage.BUILDER: [
                "turn this approach into a reusable workflow",
                "design modules and interfaces for implementation",
                "create a repeatable playbook for this task",
                "standardize the process so others can run it",
                "build automation for recurring decision steps",
                "productize this method into a durable system",
                "define components and handoffs for the pipeline",
                "write implementation specs for scalable execution",
                "convert ad hoc work into a structured template",
                "design a system architecture for this pattern",
                "establish operational SOPs for repeated use",
                "package this into a maintainable toolkit",
            ],
        }
        self._exemplar_embeddings: Dict[Stage, List[List[float]]] = {
            stage: self.model.encode(exemplars, normalize_embeddings=True).tolist()
            for stage, exemplars in self.stage_exemplars.items()
        }

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def score(self, task: str) -> EmbeddingScore:
        task_embedding = self.model.encode(task, normalize_embeddings=True).tolist()
        stage_scores: Dict[Stage, float] = {}
        for stage, exemplar_vectors in self._exemplar_embeddings.items():
            similarities = [self._cosine_similarity(task_embedding, exemplar) for exemplar in exemplar_vectors]
            stage_scores[stage] = max(similarities) if similarities else 0.0

        best_stage, best_score = max(stage_scores.items(), key=lambda item: item[1])
        return EmbeddingScore(
            stage_scores=stage_scores,
            best_stage=best_stage,
            best_score=float(best_score),
            below_threshold=best_score < 0.35,
        )
