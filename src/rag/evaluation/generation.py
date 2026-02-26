"""Generation metrics for RAG evaluation.

This module implements metrics for evaluating the quality of generated answers
in RAG systems, including semantic similarity, faithfulness, and relevance.

Metrics implemented:
    - BERTScore: Semantic similarity using BERT embeddings
    - Faithfulness: Whether answer is grounded in contexts
    - Answer Relevance: Whether answer addresses the question
    - BLEU: N-gram overlap metric (optional)
    - ROUGE: Longest common subsequence metric (optional)

Performance targets:
    - BERTScore computation: < 100ms
    - Faithfulness/Relevance: < 50ms (when using cached embeddings)
"""

import re
import time
from typing import Any, Protocol

import numpy as np


class LLMProvider(Protocol):
    """Protocol for LLM provider used in faithfulness checking."""
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> object:
        """Call LLM for completion."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding provider used in relevance scoring."""
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors."""
        ...


class GenerationMetrics:
    """Generation quality metrics for RAG evaluation.
    
    This class provides methods for evaluating the quality of generated answers
    using various metrics including semantic similarity and faithfulness.
    
    Example:
        >>> from src.rag.evaluation.generation import GenerationMetrics
        >>>
        >>> metrics = GenerationMetrics()
        >>>
        >>> # BERTScore
        >>> bert_scores = await metrics.bertscore(
        ...     answer="Machine learning is...",
        ...     ground_truth_answer="ML is part of artificial intelligence..."
        ... )
        >>>
        >>> # Faithfulness
        >>> faithfulness = await metrics.faithfulness(
        ...     answer="RAG improves accuracy",
        ...     contexts=["RAG combines retrieval with generation"]
        ... )
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        use_cache: bool = True
    ):
        """Initialize generation metrics.
        
        Args:
            llm_provider: Optional LLM provider for faithfulness checking
            embedding_provider: Optional embedding provider for relevance
            bertscore_model: Model to use for BERTScore
            use_cache: Whether to cache embedding results
        """
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.bertscore_model = bertscore_model
        self.use_cache = use_cache
        self._embedding_cache: dict[str, list[float]] = {}
        self._bertscore_scorer: object | None = None
    
    async def bertscore(
        self,
        answer: str,
        ground_truth_answer: str,
        lang: str = "en"
    ) -> dict[str, float]:
        """Calculate BERTScore for semantic similarity.
        
        BERTScore uses pre-trained BERT contextual embeddings to compute
        token-level similarity between candidate and reference texts.
        
        Args:
            answer: Generated answer text
            ground_truth_answer: Ground truth answer text
            lang: Language code (default: "en")
            
        Returns:
            Dictionary with precision, recall, and f1 scores
            
        Note:
            This method tries to use the bert-score library if available,
            otherwise falls back to a simple embedding-based similarity.
            
        Example:
            >>> scores = await metrics.bertscore(
            ...     answer="Machine learning is a subset of AI",
            ...     ground_truth_answer="ML is part of artificial intelligence"
            ... )
            >>> print(f"F1: {scores['f1']}")
        """
        start_time = time.perf_counter()
        
        try:
            # Try to use bert-score library
            from bert_score import score
            
            # bert-score expects lists
            P, R, F1 = score(
                [answer],
                [ground_truth_answer],
                lang=lang,
                model_type=self.bertscore_model,
                verbose=False,
                device="cpu"  # Use CPU for consistency
            )
            
            result = {
                "precision": float(P.mean().item()),
                "recall": float(R.mean().item()),
                "f1": float(F1.mean().item()),
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }
            
        except ImportError:
            # Fallback to simple embedding similarity
            result = await self._embedding_similarity(
                answer, ground_truth_answer
            )
            result["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def bertscore_batch(
        self,
        answers: list[str],
        ground_truth_answers: list[str],
        lang: str = "en"
    ) -> dict[str, Any]:
        """Calculate BERTScore for multiple answer pairs.
        
        Args:
            answers: List of generated answers
            ground_truth_answers: List of ground truth answers
            lang: Language code (default: "en")
            
        Returns:
            Dictionary with lists of precision, recall, f1 scores,
            plus mean values
        """
        if len(answers) != len(ground_truth_answers):
            raise ValueError("answers and ground_truth_answers must have same length")
        
        start_time = time.perf_counter()
        
        try:
            from bert_score import score
            
            P, R, F1 = score(
                answers,
                ground_truth_answers,
                lang=lang,
                model_type=self.bertscore_model,
                verbose=False,
                device="cpu"
            )
            
            result: dict[str, Any] = {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist(),
                "mean_precision": float(P.mean().item()),
                "mean_recall": float(R.mean().item()),
                "mean_f1": float(F1.mean().item()),
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }
            
        except ImportError:
            # Fallback to individual calculations
            precisions, recalls, f1s = [], [], []
            
            for ans, gt in zip(answers, ground_truth_answers):
                scores = await self._embedding_similarity(ans, gt)
                precisions.append(scores["precision"])
                recalls.append(scores["recall"])
                f1s.append(scores["f1"])
            
            result = {
                "precision": precisions,
                "recall": recalls,
                "f1": f1s,
                "mean_precision": float(np.mean(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "mean_f1": float(np.mean(f1s)),
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }
        
        return result
    
    async def _embedding_similarity(
        self,
        text1: str,
        text2: str
    ) -> dict[str, float]:
        """Compute embedding-based similarity as BERTScore fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with precision, recall, f1 (all equal to cosine similarity)
        """
        if self.embedding_provider is None:
            # Fallback to simple string similarity
            return self._simple_similarity(text1, text2)
        
        # Use embedding provider
        cache_key_1 = str(hash(text1) & 0xFFFFFFFF)
        cache_key_2 = str(hash(text2) & 0xFFFFFFFF)
        
        if self.use_cache:
            emb1 = self._embedding_cache.get(cache_key_1)
            emb2 = self._embedding_cache.get(cache_key_2)
        else:
            emb1 = emb2 = None
        
        if emb1 is None:
            embeddings = await self.embedding_provider.embed([text1])
            emb1 = embeddings[0]
            if self.use_cache:
                self._embedding_cache[cache_key_1] = emb1
        
        if emb2 is None:
            embeddings = await self.embedding_provider.embed([text2])
            emb2 = embeddings[0]
            if self.use_cache:
                self._embedding_cache[cache_key_2] = emb2
        
        # Cosine similarity
        similarity = self._cosine_similarity(emb1, emb2)
        
        # For embedding similarity, precision = recall = f1 = similarity
        return {
            "precision": similarity,
            "recall": similarity,
            "f1": similarity
        }
    
    def _simple_similarity(self, text1: str, text2: str) -> dict[str, float]:
        """Simple token-based similarity as last resort fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with precision, recall, f1
        """
        tokens1 = set(self._tokenize(text1.lower()))
        tokens2 = set(self._tokenize(text2.lower()))
        
        if not tokens1 or not tokens2:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        intersection = len(tokens1 & tokens2)
        precision = intersection / len(tokens1) if tokens1 else 0.0
        recall = intersection / len(tokens2) if tokens2 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for fallback similarity.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word tokenization
        return re.findall(r"\b\w+\b", text.lower())
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (-1 to 1, typically 0 to 1 for embeddings)
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    async def faithfulness(
        self,
        answer: str,
        contexts: list[str],
        method: str = "nli"
    ) -> float:
        """Calculate faithfulness score.
        
        Faithfulness measures whether the generated answer is grounded in
        and supported by the provided contexts. A high faithfulness score
        indicates the answer doesn't hallucinate or contradict the contexts.
        
        Args:
            answer: Generated answer text
            contexts: List of retrieved context texts
            method: Method to use - "nli" (NLI model) or "token_overlap"
            
        Returns:
            Faithfulness score between 0 and 1
            
        Example:
            >>> faithfulness = await metrics.faithfulness(
            ...     answer="RAG combines retrieval with generation",
            ...     contexts=["RAG retrieves documents and uses them for generation"]
            ... )
        """
        if not contexts:
            return 0.0
        
        if method == "nli" and self.llm_provider:
            return await self._faithfulness_nli(answer, contexts)
        else:
            return self._faithfulness_token_overlap(answer, contexts)
    
    async def _faithfulness_nli(
        self,
        answer: str,
        contexts: list[str]
    ) -> float:
        """Calculate faithfulness using NLI (Natural Language Inference).
        
        Uses an LLM to check if the answer is entailed by the contexts.
        
        Args:
            answer: Generated answer text
            contexts: List of context texts
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not self.llm_provider:
            return self._faithfulness_token_overlap(answer, contexts)
        
        # Combine contexts
        combined_context = " ".join(contexts)
        
        # Create NLI prompt
        prompt = f"""Given the following context, determine if the answer is supported.

Context: {combined_context[:1000]}

Answer: {answer[:500]}

Is the answer fully supported by the context? Rate from 0 to 1, where:
- 1.0 = Fully supported, all claims are in the context
- 0.5 = Partially supported, some claims are in the context
- 0.0 = Not supported, claims contradict or are not in the context

Respond with ONLY a number between 0 and 1."""
        
        try:
            response = await self.llm_provider.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            content = response.content if hasattr(response, "content") else str(response)
            
            # Extract number from response
            numbers = re.findall(r"0?\.\d+|1\.0|1|0", content.strip())
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            
        except Exception:
            pass
        
        # Fallback to token overlap
        return self._faithfulness_token_overlap(answer, contexts)
    
    def _faithfulness_token_overlap(
        self,
        answer: str,
        contexts: list[str]
    ) -> float:
        """Calculate faithfulness using token overlap.
        
        A simple heuristic that measures content overlap between answer
        and contexts as a proxy for faithfulness.
        
        Args:
            answer: Generated answer text
            contexts: List of context texts
            
        Returns:
            Faithfulness score between 0 and 1
        """
        answer_tokens = set(self._tokenize(answer.lower()))
        
        if not answer_tokens:
            return 1.0  # Empty answer is trivially faithful
        
        # Combine all context tokens
        all_context_tokens: set[str] = set()
        for context in contexts:
            all_context_tokens.update(self._tokenize(context.lower()))
        
        if not all_context_tokens:
            return 0.0
        
        # Calculate overlap
        overlap = len(answer_tokens & all_context_tokens)
        coverage = overlap / len(answer_tokens)
        
        # Also consider if answer contains key facts from context
        # (we could be more sophisticated here)
        return min(1.0, coverage)
    
    async def answer_relevance(
        self,
        answer: str,
        question: str
    ) -> float:
        """Calculate answer relevance score.
        
        Relevance measures whether the answer actually addresses the question.
        Uses semantic similarity between the question and answer.
        
        Args:
            answer: Generated answer text
            question: Original question/query
            
        Returns:
            Relevance score between 0 and 1
            
        Example:
            >>> relevance = await metrics.answer_relevance(
            ...     answer="Machine learning is a subset of AI...",
            ...     question="What is machine learning?"
            ... )
        """
        if not answer or not question:
            return 0.0
        
        if self.embedding_provider:
            # Use semantic similarity with embeddings
            similarity_scores = await self._embedding_similarity(question, answer)
            return similarity_scores["f1"]
        else:
            # Use simple token overlap
            q_tokens = set(self._tokenize(question.lower()))
            a_tokens = set(self._tokenize(answer.lower()))
            
            if not q_tokens:
                return 1.0 if a_tokens else 0.0
            
            overlap = len(q_tokens & a_tokens)
            return min(1.0, overlap / len(q_tokens))
    
    def bleu_score(
        self,
        answer: str,
        ground_truth_answer: str,
        max_n: int = 4
    ) -> float:
        """Calculate BLEU score for n-gram overlap.
        
        BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
        between candidate and reference texts.
        
        Args:
            answer: Generated answer text
            ground_truth_answer: Ground truth answer text
            max_n: Maximum n-gram size (default: 4)
            
        Returns:
            BLEU score between 0 and 1
            
        Note:
            This is a simplified implementation. For production use,
            consider using NLTK or sacrebleu for more accurate scores.
        """
        try:
            # Try to use nltk
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
            
            reference = self._tokenize(ground_truth_answer.lower())
            candidate = self._tokenize(answer.lower())
            
            smoothing = SmoothingFunction().method1
            weights = [1.0 / max_n] * max_n
            
            score = sentence_bleu(
                [reference],
                candidate,
                weights=weights,
                smoothing_function=smoothing
            )
            
            return float(score)
            
        except ImportError:
            # Fallback to simple n-gram overlap
            return self._simple_ngram_overlap(
                answer, ground_truth_answer, max_n=max_n
            )
    
    def rouge_l_score(
        self,
        answer: str,
        ground_truth_answer: str
    ) -> dict[str, float]:
        """Calculate ROUGE-L score (Longest Common Subsequence).
        
        ROUGE-L measures the longest common subsequence between candidate
        and reference texts, which captures sentence-level structure.
        
        Args:
            answer: Generated answer text
            ground_truth_answer: Ground truth answer text
            
        Returns:
            Dictionary with precision, recall, and f1 scores
            
        Note:
            This is a simplified implementation. For production use,
            consider using rouge-score package.
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(ground_truth_answer, answer)
            
            return {
                "precision": float(scores["rougeL"].precision),
                "recall": float(scores["rougeL"].recall),
                "f1": float(scores["rougeL"].fmeasure)
            }
            
        except ImportError:
            # Fallback to simple LCS
            return self._simple_rouge_l(answer, ground_truth_answer)
    
    def _simple_ngram_overlap(
        self,
        text1: str,
        text2: str,
        max_n: int = 4
    ) -> float:
        """Simple n-gram overlap calculation.
        
        Args:
            text1: First text
            text2: Second text
            max_n: Maximum n-gram size
            
        Returns:
            N-gram overlap score
        """
        tokens1 = self._tokenize(text1.lower())
        tokens2 = self._tokenize(text2.lower())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        scores = []
        for n in range(1, max_n + 1):
            ngrams1 = set(self._get_ngrams(tokens1, n))
            ngrams2 = set(self._get_ngrams(tokens2, n))
            
            if ngrams1 and ngrams2:
                overlap = len(ngrams1 & ngrams2)
                precision = overlap / len(ngrams1)
                recall = overlap / len(ngrams2)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores.append(f1)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple[str, ...]]:
        """Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams as tuples
        """
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    def _simple_rouge_l(
        self,
        text1: str,
        text2: str
    ) -> dict[str, float]:
        """Simple ROUGE-L calculation using LCS.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with precision, recall, f1
        """
        tokens1 = self._tokenize(text1.lower())
        tokens2 = self._tokenize(text2.lower())
        
        if not tokens1 or not tokens2:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        lcs_length = self._lcs_length(tokens1, tokens2)
        
        precision = lcs_length / len(tokens1) if tokens1 else 0.0
        recall = lcs_length / len(tokens2) if tokens2 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate length of Longest Common Subsequence.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)
        
        # Use dynamic programming with space optimization
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev
        
        return prev[n]
    
    async def compute_all_metrics(
        self,
        answer: str,
        ground_truth_answer: str,
        question: str,
        contexts: list[str],
        include_bleu_rouge: bool = False
    ) -> dict[str, float]:
        """Compute all generation metrics at once.
        
        This is an optimized method that computes multiple metrics efficiently.
        
        Args:
            answer: Generated answer text
            ground_truth_answer: Ground truth answer text
            question: Original question
            contexts: List of context texts
            include_bleu_rouge: Whether to include BLEU and ROUGE scores
            
        Returns:
            Dictionary containing all computed metrics
        """
        start_time = time.perf_counter()
        
        metrics: dict[str, float] = {}
        
        # BERTScore
        bert_scores = await self.bertscore(answer, ground_truth_answer)
        metrics["bertscore_precision"] = bert_scores["precision"]
        metrics["bertscore_recall"] = bert_scores["recall"]
        metrics["bertscore_f1"] = bert_scores["f1"]
        
        # Faithfulness
        metrics["faithfulness"] = await self.faithfulness(answer, contexts)
        
        # Answer Relevance
        metrics["answer_relevance"] = await self.answer_relevance(answer, question)
        
        # Optional BLEU and ROUGE
        if include_bleu_rouge:
            metrics["bleu"] = self.bleu_score(answer, ground_truth_answer)
            rouge_scores = self.rouge_l_score(answer, ground_truth_answer)
            metrics["rouge_l_f1"] = rouge_scores["f1"]
        
        metrics["latency_ms"] = (time.perf_counter() - start_time) * 1000
        
        return metrics
