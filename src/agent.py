import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any
import httpx
import numpy as np
from pydantic import ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from schemas import (
    EvalRequest, EvalConfig, Query, QueryRequest, 
    RetrievalResponse, QueryResult, EvaluationSummary, EvaluationReport
)
from metrics import (
    calculate_ndcg_at_k, calculate_mrr_at_k, 
    calculate_precision_at_k, calculate_recall_at_k,
    count_relevant_retrieved, count_total_relevant
)


class Agent:
    required_roles: list[str] = ["retrieval_agent"]
    required_config_keys: list[str] = ["num_queries", "top_k", "random_seed"]

    def __init__(self):
        self.messenger = Messenger()
        self.all_queries = self.load_queries()
        self.qrels = self.load_qrels()

    def load_queries(self) -> List[Query]:
        """Load all test queries from data file."""
        queries_file = Path("data/test_queries.jsonl")
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                queries.append(Query(**data))
        return queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments structured as dict[query_id, dict[doc_id, score]]."""
        qrels_file = Path("data/test_qrels.jsonl")
        if not qrels_file.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_file}")
        
        qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                query_id = data['query-id']
                doc_id = data['corpus-id']
                score = data['score']
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        
        return qrels

    def sample_queries(self, num_queries: int, random_seed: int) -> List[Query]:
        """Sample queries randomly with given seed for reproducibility."""
        random.seed(random_seed)
        
        valid_queries = [q for q in self.all_queries if q.id in self.qrels]
        
        sample_size = min(num_queries, len(valid_queries))
        return random.sample(valid_queries, sample_size)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the evaluation request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        try:
            EvalConfig(**request.config)
        except ValidationError as e:
            return False, f"Invalid config: {e}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run the NFCorpus retrieval evaluation."""
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        config = EvalConfig(**request.config)
        purple_agent_url = str(request.participants["retrieval_agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation with {config.num_queries} queries (seed: {config.random_seed})")
        )

        sampled_queries = self.sample_queries(config.num_queries, config.random_seed)
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sampled {len(sampled_queries)} queries. Evaluating purple agent...")
        )

        query_results = []
        successful_count = 0
        failed_count = 0
        qdrant_calls_total = 0
        mcp_qdrant_calls_before = None
        start_time = time.time()
        max_query_time = 60.0  # Maximum 60 seconds per query

        if config.mcp_server_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(f"{config.mcp_server_url.rstrip('/')}/health_check", json={})
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict) and isinstance(data.get("qdrant_calls_total"), int):
                        mcp_qdrant_calls_before = data["qdrant_calls_total"]
            except Exception:
                mcp_qdrant_calls_before = None
        
        for idx, query in enumerate(sampled_queries, 1):
            if idx % 10 == 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Progress: {idx}/{len(sampled_queries)} queries evaluated")
                )
            
            try:
                query_start = time.time()
                query_request = QueryRequest(query=query.text, top_k=config.top_k)
                response_text = await self.messenger.talk_to_agent(
                    message=query_request.model_dump_json(),
                    url=purple_agent_url,
                    timeout=max_query_time
                )
                query_elapsed = time.time() - query_start
                
                # Warn if query took too long
                if query_elapsed > max_query_time * 0.9:  # Warn at 90% of limit
                    print(f"WARNING: Query {query.id} took {query_elapsed:.1f}s (close to {max_query_time}s limit)")
                
                try:
                    response_data = json.loads(response_text)
                    retrieval_response = RetrievalResponse(**response_data)
                except Exception:
                    retrieval_response = RetrievalResponse(doc_ids=[])
                
                ground_truth = self.qrels.get(query.id, {})
                
                ndcg = calculate_ndcg_at_k(
                    retrieval_response.doc_ids,
                    ground_truth,
                    k=config.top_k
                )
                mrr = calculate_mrr_at_k(
                    retrieval_response.doc_ids,
                    ground_truth,
                    k=config.top_k
                )
                precision = calculate_precision_at_k(
                    retrieval_response.doc_ids,
                    ground_truth,
                    k=config.top_k
                )
                recall = calculate_recall_at_k(
                    retrieval_response.doc_ids,
                    ground_truth,
                    k=config.top_k
                )
                
                query_result = QueryResult(
                    query_id=query.id,
                    query_text=query.text,
                    retrieved_docs=retrieval_response.doc_ids[:config.top_k],
                    ndcg_at_5=ndcg,
                    mrr_at_5=mrr,
                    precision_at_5=precision,
                    recall_at_5=recall,
                    relevant_docs_retrieved=count_relevant_retrieved(
                        retrieval_response.doc_ids, ground_truth
                    ),
                    total_relevant_docs=count_total_relevant(ground_truth)
                )
                query_results.append(query_result)
                successful_count += 1
                
            except Exception:
                query_result = QueryResult(
                    query_id=query.id,
                    query_text=query.text,
                    retrieved_docs=[],
                    ndcg_at_5=0.0,
                    mrr_at_5=0.0,
                    precision_at_5=0.0,
                    recall_at_5=0.0,
                    relevant_docs_retrieved=0,
                    total_relevant_docs=count_total_relevant(self.qrels.get(query.id, {}))
                )
                query_results.append(query_result)
                failed_count += 1

        total_time = time.time() - start_time

        if config.mcp_server_url and mcp_qdrant_calls_before is not None:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(f"{config.mcp_server_url.rstrip('/')}/health_check", json={})
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict) and isinstance(data.get("qdrant_calls_total"), int):
                        mcp_after = data["qdrant_calls_total"]
                        qdrant_calls_total = max(0, mcp_after - mcp_qdrant_calls_before)
            except Exception:
                qdrant_calls_total = 0
        else:
            qdrant_calls_total = 0
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Calculating aggregate metrics...")
        )

        ndcg_scores = [r.ndcg_at_5 for r in query_results]
        mrr_scores = [r.mrr_at_5 for r in query_results]
        precision_scores = [r.precision_at_5 for r in query_results]
        recall_scores = [r.recall_at_5 for r in query_results]
        
        queries_per_second = len(query_results) / total_time if total_time > 0 else 0.0
        
        summary = EvaluationSummary(
            total_queries=len(query_results),
            successful_queries=successful_count,
            failed_queries=failed_count,
            qdrant_calls_total=qdrant_calls_total,
            mean_ndcg_at_5=float(np.mean(ndcg_scores)),
            mean_mrr_at_5=float(np.mean(mrr_scores)),
            mean_precision_at_5=float(np.mean(precision_scores)),
            mean_recall_at_5=float(np.mean(recall_scores)),
            median_ndcg_at_5=float(np.median(ndcg_scores)),
            std_ndcg_at_5=float(np.std(ndcg_scores)),
            min_ndcg_at_5=float(np.min(ndcg_scores)),
            max_ndcg_at_5=float(np.max(ndcg_scores)),
            success_rate=sum(1 for s in ndcg_scores if s > 0) / len(ndcg_scores),
            total_time_seconds=total_time,
            queries_per_second=queries_per_second
        )

        report = EvaluationReport(
            summary=summary,
            per_query_results=query_results,
            config=config,
            random_seed_used=config.random_seed
        )

        summary_text = f"""NFCorpus Retrieval Evaluation Results
{'='*60}
Total Queries: {summary.total_queries}
Successful: {summary.successful_queries}
Failed: {summary.failed_queries}
Qdrant Calls: {summary.qdrant_calls_total}
{'='*60}
Mean NDCG@5: {summary.mean_ndcg_at_5:.4f}
Mean MRR@5: {summary.mean_mrr_at_5:.4f}
Mean Precision@5: {summary.mean_precision_at_5:.4f}
Mean Recall@5: {summary.mean_recall_at_5:.4f}
Median NDCG@5: {summary.median_ndcg_at_5:.4f}
Std NDCG@5: {summary.std_ndcg_at_5:.4f}
Min NDCG@5: {summary.min_ndcg_at_5:.4f}
Max NDCG@5: {summary.max_ndcg_at_5:.4f}
Success Rate: {summary.success_rate:.2%}
{'='*60}
Performance:
  - Total Time: {summary.total_time_seconds:.1f}s
  - Queries/Second: {summary.queries_per_second:.2f}
{'='*60}
Configuration:
  - Queries: {config.num_queries}
  - Top-K: {config.top_k}
  - Random Seed: {config.random_seed}
"""

        output_data = {
            "mean_ndcg_at_k": round(summary.mean_ndcg_at_5, 2),
            "mean_mrr_at_k": round(summary.mean_mrr_at_5, 2),
            "mean_precision_at_k": round(summary.mean_precision_at_5, 2),
            "mean_recall_at_k": round(summary.mean_recall_at_5, 2),
            "execution_time_seconds": round(summary.total_time_seconds, 2),
            "total_queries": summary.total_queries,
            "successful_communications": summary.successful_queries,
            "vector_search_calls_total": summary.qdrant_calls_total,
        }
        
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=output_data))
            ],
            name="Evaluation Results",
        )
