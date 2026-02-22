"""Unit tests for generation metrics.

Tests all generation metric computations including BERTScore, faithfulness,
answer relevance, BLEU, and ROUGE.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.evaluation.generation import GenerationMetrics


class TestGenerationMetricsInitialization:
    """Tests for GenerationMetrics initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default values."""
        metrics = GenerationMetrics()
        
        assert metrics.llm_provider is None
        assert metrics.embedding_provider is None
        assert metrics.bertscore_model == "microsoft/deberta-xlarge-mnli"
        assert metrics.use_cache is True
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        mock_llm = MagicMock()
        mock_emb = MagicMock()
        
        metrics = GenerationMetrics(
            llm_provider=mock_llm,
            embedding_provider=mock_emb,
            bertscore_model="custom-model",
            use_cache=False
        )
        
        assert metrics.llm_provider == mock_llm
        assert metrics.embedding_provider == mock_emb
        assert metrics.bertscore_model == "custom-model"
        assert metrics.use_cache is False


class TestGenerationMetricsBERTScore:
    """Tests for BERTScore computation."""
    
    @pytest.mark.asyncio
    async def test_bertscore_identical_texts(self):
        """Test BERTScore with identical texts."""
        metrics = GenerationMetrics()
        
        result = await metrics.bertscore(
            answer="This is a test sentence.",
            ground_truth_answer="This is a test sentence."
        )
        
        # All scores should be high (close to 1.0) for identical texts
        assert result["precision"] > 0.9
        assert result["recall"] > 0.9
        assert result["f1"] > 0.9
        assert "latency_ms" in result
    
    @pytest.mark.asyncio
    async def test_bertscore_different_texts(self):
        """Test BERTScore with different but related texts."""
        metrics = GenerationMetrics()
        
        result = await metrics.bertscore(
            answer="Machine learning is a subset of AI.",
            ground_truth_answer="ML is part of artificial intelligence."
        )
        
        # Scores should be moderate for semantically similar texts (using token overlap fallback)
        assert result["precision"] > 0.2
        assert result["recall"] > 0.2
        assert result["f1"] > 0.2
    
    @pytest.mark.asyncio
    async def test_bertscore_very_different_texts(self):
        """Test BERTScore with completely different texts."""
        metrics = GenerationMetrics()
        
        result = await metrics.bertscore(
            answer="The weather is nice today.",
            ground_truth_answer="Machine learning is a subset of AI."
        )
        
        # Scores should be low for unrelated texts
        assert result["f1"] < 0.7
    
    @pytest.mark.asyncio
    async def test_bertscore_empty_texts(self):
        """Test BERTScore with empty texts."""
        metrics = GenerationMetrics()
        
        result = await metrics.bertscore(
            answer="",
            ground_truth_answer="Some text."
        )
        
        # Should handle gracefully
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result


class TestGenerationMetricsBERTScoreBatch:
    """Tests for batch BERTScore computation."""
    
    @pytest.mark.asyncio
    async def test_bertscore_batch(self):
        """Test batch BERTScore computation."""
        metrics = GenerationMetrics()
        
        answers = ["Text one.", "Text two."]
        ground_truths = ["Text one.", "Different text."]
        
        result = await metrics.bertscore_batch(answers, ground_truths)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "mean_precision" in result
        assert "mean_recall" in result
        assert "mean_f1" in result
        assert len(result["f1"]) == 2
    
    def test_bertscore_batch_mismatched_lengths(self):
        """Test batch BERTScore with mismatched list lengths."""
        metrics = GenerationMetrics()
        
        with pytest.raises(ValueError, match="same length"):
            # Use synchronous call since the error is raised immediately
            import asyncio
            asyncio.run(metrics.bertscore_batch(
                ["text1", "text2"],
                ["text1"]
            ))


class TestGenerationMetricsFaithfulness:
    """Tests for faithfulness computation."""
    
    @pytest.mark.asyncio
    async def test_faithfulness_high_overlap(self):
        """Test faithfulness with high context overlap."""
        metrics = GenerationMetrics()
        
        answer = "RAG combines retrieval with generation."
        contexts = ["RAG retrieves documents and uses them for generation."]
        
        score = await metrics.faithfulness(answer, contexts)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Token overlap is lower than semantic similarity
    
    @pytest.mark.asyncio
    async def test_faithfulness_low_overlap(self):
        """Test faithfulness with low context overlap."""
        metrics = GenerationMetrics()
        
        answer = "The weather is sunny today."
        contexts = ["RAG retrieves documents for generation."]
        
        score = await metrics.faithfulness(answer, contexts)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be fairly low
    
    @pytest.mark.asyncio
    async def test_faithfulness_empty_contexts(self):
        """Test faithfulness with empty contexts."""
        metrics = GenerationMetrics()
        
        score = await metrics.faithfulness("Some answer.", [])
        
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_faithfulness_multiple_contexts(self):
        """Test faithfulness with multiple contexts."""
        metrics = GenerationMetrics()
        
        answer = "RAG improves accuracy and reduces hallucinations."
        contexts = [
            "RAG retrieves relevant documents.",
            "RAG reduces hallucinations by grounding in facts.",
        ]
        
        score = await metrics.faithfulness(answer, contexts)
        
        assert 0.0 <= score <= 1.0


class TestGenerationMetricsAnswerRelevance:
    """Tests for answer relevance computation."""
    
    @pytest.mark.asyncio
    async def test_relevance_high_match(self):
        """Test relevance with high question-answer match."""
        metrics = GenerationMetrics()
        
        answer = "Machine learning is a subset of AI that enables computers to learn."
        question = "What is machine learning?"
        
        score = await metrics.answer_relevance(answer, question)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Token overlap is lower than semantic similarity
    
    @pytest.mark.asyncio
    async def test_relevance_low_match(self):
        """Test relevance with low question-answer match."""
        metrics = GenerationMetrics()
        
        answer = "The weather is nice today."
        question = "What is machine learning?"
        
        score = await metrics.answer_relevance(answer, question)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be fairly low
    
    @pytest.mark.asyncio
    async def test_relevance_empty_question(self):
        """Test relevance with empty question."""
        metrics = GenerationMetrics()
        
        score = await metrics.answer_relevance("Some answer.", "")
        
        assert score == 0.0


class TestGenerationMetricsBLEU:
    """Tests for BLEU score computation."""
    
    def test_bleu_identical(self):
        """Test BLEU with identical texts."""
        metrics = GenerationMetrics()
        
        score = metrics.bleu_score(
            "This is a test.",
            "This is a test."
        )
        
        assert score == 1.0
    
    def test_bleu_different(self):
        """Test BLEU with different texts."""
        metrics = GenerationMetrics()
        
        score = metrics.bleu_score(
            "Machine learning is great.",
            "The weather is nice."
        )
        
        assert 0.0 <= score < 1.0
    
    def test_bleu_partial_overlap(self):
        """Test BLEU with partial overlap."""
        metrics = GenerationMetrics()
        
        score = metrics.bleu_score(
            "Machine learning is a subset of AI.",
            "Machine learning is part of artificial intelligence."
        )
        
        assert 0.0 < score < 1.0


class TestGenerationMetricsROUGE:
    """Tests for ROUGE-L score computation."""
    
    def test_rouge_identical(self):
        """Test ROUGE-L with identical texts."""
        metrics = GenerationMetrics()
        
        scores = metrics.rouge_l_score(
            "This is a test.",
            "This is a test."
        )
        
        assert scores["f1"] == 1.0
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
    
    def test_rouge_different(self):
        """Test ROUGE-L with different texts."""
        metrics = GenerationMetrics()
        
        scores = metrics.rouge_l_score(
            "Machine learning is great.",
            "The weather is nice."
        )
        
        assert 0.0 <= scores["f1"] < 1.0
    
    def test_rouge_partial_overlap(self):
        """Test ROUGE-L with partial overlap."""
        metrics = GenerationMetrics()
        
        scores = metrics.rouge_l_score(
            "Machine learning is a subset of AI.",
            "Machine learning is part of artificial intelligence."
        )
        
        assert 0.0 < scores["f1"] <= 1.0


class TestGenerationMetricsComputeAll:
    """Tests for compute_all_metrics method."""
    
    @pytest.mark.asyncio
    async def test_compute_all_returns_all_metrics(self):
        """Test that compute_all returns all expected metrics."""
        metrics = GenerationMetrics()
        
        result = await metrics.compute_all_metrics(
            answer="Machine learning is a subset of AI.",
            ground_truth_answer="ML is part of AI.",
            question="What is ML?",
            contexts=["ML is machine learning."],
            include_bleu_rouge=True
        )
        
        assert "bertscore_precision" in result
        assert "bertscore_recall" in result
        assert "bertscore_f1" in result
        assert "faithfulness" in result
        assert "answer_relevance" in result
        assert "bleu" in result
        assert "rouge_l_f1" in result
        assert "latency_ms" in result
    
    @pytest.mark.asyncio
    async def test_compute_all_without_bleu_rouge(self):
        """Test compute_all without BLEU/ROUGE."""
        metrics = GenerationMetrics()
        
        result = await metrics.compute_all_metrics(
            answer="Test answer.",
            ground_truth_answer="Test answer.",
            question="What is test?",
            contexts=["Test context."],
            include_bleu_rouge=False
        )
        
        assert "bertscore_f1" in result
        assert "bleu" not in result
        assert "rouge_l_f1" not in result


class TestGenerationMetricsUtilityMethods:
    """Tests for utility methods."""
    
    def test_tokenize(self):
        """Test tokenization."""
        metrics = GenerationMetrics()
        
        tokens = metrics._tokenize("Hello world!")
        
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        metrics = GenerationMetrics()
        
        vec = [1.0, 2.0, 3.0]
        similarity = metrics._cosine_similarity(vec, vec)
        
        assert similarity == pytest.approx(1.0, abs=0.001)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        metrics = GenerationMetrics()
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = metrics._cosine_similarity(vec1, vec2)
        
        assert similarity == pytest.approx(0.0, abs=0.001)
    
    def test_simple_similarity(self):
        """Test simple similarity fallback."""
        metrics = GenerationMetrics()
        
        result = metrics._simple_similarity(
            "machine learning ai",
            "machine learning"
        )
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert 0.0 < result["f1"] <= 1.0


class TestGenerationMetricsNLI:
    """Tests for NLI-based faithfulness."""
    
    @pytest.mark.asyncio
    async def test_faithfulness_nli_with_llm(self):
        """Test faithfulness with LLM provider."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "0.9"
        mock_llm.chat_completion = AsyncMock(return_value=mock_response)
        
        metrics = GenerationMetrics(llm_provider=mock_llm)
        
        score = await metrics.faithfulness(
            answer="RAG improves accuracy.",
            contexts=["RAG retrieves documents for generation."],
            method="nli"
        )
        
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.9, abs=0.01)
    
    @pytest.mark.asyncio
    async def test_faithfulness_nli_fallback(self):
        """Test NLI fallback when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(side_effect=Exception("LLM error"))
        
        metrics = GenerationMetrics(llm_provider=mock_llm)
        
        score = await metrics.faithfulness(
            answer="RAG improves accuracy.",
            contexts=["RAG retrieves documents."],
            method="nli"
        )
        
        # Should fall back to token overlap
        assert 0.0 <= score <= 1.0


class TestGenerationMetricsPerformance:
    """Tests for performance requirements."""
    
    @pytest.mark.asyncio
    async def test_bertscore_performance(self):
        """Test BERTScore computation time (< 100ms target)."""
        import time
        
        metrics = GenerationMetrics()
        
        start = time.perf_counter()
        result = await metrics.bertscore(
            answer="Machine learning is a subset of AI that enables computers to learn from data.",
            ground_truth_answer="ML is a branch of AI that allows computers to learn patterns from data."
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Note: Without the actual bert-score library, this test uses fallback
        # The fallback is faster, so this should easily pass
        assert result["latency_ms"] < 100.0, f"BERTScore took {result['latency_ms']:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_faithfulness_performance(self):
        """Test faithfulness computation time (< 50ms target without LLM)."""
        import time
        
        metrics = GenerationMetrics()
        
        start = time.perf_counter()
        result = await metrics.faithfulness(
            answer="RAG combines retrieval with generation to improve accuracy.",
            contexts=[
                "RAG retrieves relevant documents before generation.",
                "This grounding improves accuracy and reduces hallucinations."
            ]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 50.0, f"Faithfulness took {elapsed_ms:.2f}ms"
