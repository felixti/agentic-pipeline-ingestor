# Spec: RAG Evaluation Framework

## Overview
Implement comprehensive evaluation metrics and tools to systematically measure retrieval and generation quality for continuous improvement.

## Requirements

### Functional Requirements
1. Retrieval metrics (MRR, NDCG, Recall@K, Precision@K)
2. Generation metrics (BLEU, ROUGE, BERTScore)
3. End-to-end evaluation with ground truth
4. A/B testing framework for strategy comparison
5. Continuous monitoring and alerting

### Evaluation Metrics

#### Retrieval Metrics
```python
class RetrievalMetrics:
    @staticmethod
    def mrr(results: list[Result], ground_truth: list[str]) -> float:
        """Mean Reciprocal Rank."""
        for i, result in enumerate(results, 1):
            if result.id in ground_truth:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg(results: list[Result], ground_truth: list[str], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain."""
        dcg = sum(
            (1.0 / np.log2(i + 2)) if r.id in ground_truth else 0
            for i, r in enumerate(results[:k])
        )
        
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    @staticmethod
    def recall_at_k(results: list[Result], ground_truth: list[str], k: int = 10) -> float:
        """Recall at K."""
        retrieved = set(r.id for r in results[:k])
        relevant = set(ground_truth)
        return len(retrieved & relevant) / len(relevant) if relevant else 0.0
    
    @staticmethod
    def precision_at_k(results: list[Result], ground_truth: list[str], k: int = 10) -> float:
        """Precision at K."""
        retrieved = set(r.id for r in results[:k])
        relevant = set(ground_truth)
        return len(retrieved & relevant) / k if k > 0 else 0.0
```

#### Generation Metrics
```python
class GenerationMetrics:
    @staticmethod
    def bertscore(predictions: list[str], references: list[str]) -> dict:
        """BERTScore for semantic similarity."""
        from bert_score import score
        
        P, R, F1 = score(predictions, references, lang="en")
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    
    @staticmethod
    def faithfulness(answer: str, contexts: list[str]) -> float:
        """Check if answer is grounded in contexts."""
        # Use NLI model to check entailment
        pass
    
    @staticmethod
    def answer_relevance(answer: str, question: str) -> float:
        """Check if answer addresses the question."""
        # Use embedding similarity
        pass
```

## API Design

```python
class RAGEvaluator:
    def __init__(self):
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.db = EvaluationDatabase()
    
    async def evaluate_retrieval(
        self,
        query: str,
        results: list[Result],
        ground_truth: list[str]
    ) -> RetrievalEvaluation:
        """Evaluate retrieval quality."""
        return RetrievalEvaluation(
            query=query,
            mrr=self.retrieval_metrics.mrr(results, ground_truth),
            ndcg_at_10=self.retrieval_metrics.ndcg(results, ground_truth, k=10),
            recall_at_5=self.retrieval_metrics.recall_at_k(results, ground_truth, k=5),
            precision_at_5=self.retrieval_metrics.precision_at_k(results, ground_truth, k=5)
        )
    
    async def evaluate_generation(
        self,
        query: str,
        answer: str,
        ground_truth_answer: str,
        contexts: list[str]
    ) -> GenerationEvaluation:
        """Evaluate generation quality."""
        bert_scores = self.generation_metrics.bertscore([answer], [ground_truth_answer])
        
        return GenerationEvaluation(
            query=query,
            bertscore_f1=bert_scores["f1"],
            faithfulness=self.generation_metrics.faithfulness(answer, contexts),
            relevance=self.generation_metrics.answer_relevance(answer, query)
        )
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        rag_system: RAGSystem
    ) -> BenchmarkResult:
        """Run full benchmark evaluation."""
        dataset = await self.load_benchmark(benchmark_name)
        
        results = []
        for item in dataset:
            # Run RAG
            rag_result = await rag_system.query(item.query)
            
            # Evaluate
            retrieval_eval = await self.evaluate_retrieval(
                item.query,
                rag_result.retrieved_chunks,
                item.relevant_chunks
            )
            
            generation_eval = await self.evaluate_generation(
                item.query,
                rag_result.answer,
                item.ground_truth_answer,
                [c.content for c in rag_result.retrieved_chunks]
            )
            
            results.append({
                "retrieval": retrieval_eval,
                "generation": generation_eval
            })
        
        return self.aggregate_results(results)
```

## Configuration
```yaml
evaluation:
  enabled: true
  
  # Automatic evaluation on queries
  auto_evaluate:
    enabled: true
    sample_rate: 0.1  # 10% of queries
    store_results: true
  
  # Metrics to compute
  metrics:
    retrieval:
      - mrr
      - ndcg@10
      - recall@5
      - precision@5
      - hit_rate@10
    
    generation:
      - bertscore
      - faithfulness
      - answer_relevance
      - latency
  
  # Benchmarks
  benchmarks:
    - name: "ms_marco"
      dataset: "microsoft/ms_marco"
    - name: "custom_qa"
      dataset: "s3://eval-data/custom-qa.json"
  
  # Alerting
  alerting:
    enabled: true
    thresholds:
      mrr_drop: 0.05
      latency_p95: 1000  # ms
    
    channels:
      - slack
      - email
```

## Database Schema
```sql
-- Evaluation results
CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id),
    retrieval_mrr FLOAT,
    retrieval_ndcg FLOAT,
    generation_bertscore FLOAT,
    generation_faithfulness FLOAT,
    latency_ms INTEGER,
    strategy_config JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark runs
CREATE TABLE benchmark_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_name VARCHAR(100),
    config JSONB,
    aggregate_metrics JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_eval_results_created ON evaluation_results(created_at);
CREATE INDEX idx_eval_results_query ON evaluation_results(query_id);
```

## Acceptance Criteria
- [ ] All retrieval metrics implemented
- [ ] All generation metrics implemented
- [ ] Benchmark framework runs successfully
- [ ] A/B testing compares strategies
- [ ] Alerts trigger on threshold violations

## Performance Expectations
| Metric Computation | Time |
|-------------------|------|
| Retrieval metrics | <10ms |
| BERTScore | <100ms |
| Full benchmark | <10min |

## Dependencies
- bert-score library
- NLTK (BLEU, ROUGE)
- Evaluation datasets (MS MARCO, etc.)
- Monitoring infrastructure
