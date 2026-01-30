import json
import time
import os
import numpy as np
import httpx
from pydantic import ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from data_loader import load_nfcorpus_dataset, sample_queries
from schemas import RetrievalQuery, RetrievalResponse, EvalRequest


def compute_ndcg(relevances: list[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(relevances: list[int], k: int) -> float:
    """Compute Mean Reciprocal Rank at k."""
    relevances = relevances[:k]
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision(relevances: list[int], k: int) -> float:
    """Compute Precision at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(1 for rel in relevances if rel > 0) / len(relevances)


def compute_recall(relevances: list[int], k: int, total_relevant: int) -> float:
    """Compute Recall at k."""
    if total_relevant == 0:
        return 0.0
    relevances = relevances[:k]
    return sum(1 for rel in relevances if rel > 0) / total_relevant


class Agent:
    required_roles: list[str] = ["retrieval_agent"]
    required_config_keys: list[str] = ["num_queries", "top_k"]

    def __init__(self):
        self.messenger = Messenger()
        self.corpus = None
        self.queries = None
        self.qrels = None

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        num_queries = request.config.get("num_queries")
        if not isinstance(num_queries, int) or num_queries <= 0:
            return False, "num_queries must be a positive integer"
        if num_queries > 323:
            return False, "num_queries must be <= 323"

        top_k = request.config.get("top_k")
        if not isinstance(top_k, int) or top_k <= 0:
            return False, "top_k must be a positive integer"
        if top_k > 100:
            return False, "top_k must be <= 100"

        random_seed = request.config.get("random_seed")
        if random_seed is not None and not isinstance(random_seed, int):
            return False, "random_seed must be an integer"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Evaluate purple agent on NFCorpus retrieval task.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        purple_agent_url = str(request.participants["retrieval_agent"])
        num_queries = request.config["num_queries"]
        top_k = request.config["top_k"]
        random_seed = request.config.get("random_seed")

        await updater.update_status(
            TaskState.working, new_agent_text_message("Loading NFCorpus dataset...")
        )

        try:
            if self.corpus is None or self.queries is None or self.qrels is None:
                self.corpus, self.queries, self.qrels = load_nfcorpus_dataset()
        except Exception as e:
            await updater.reject(new_agent_text_message(f"Failed to load dataset: {e}"))
            return

        sampled_queries = sample_queries(self.queries, num_queries, random_seed, self.qrels)

        # Reset MCP server stats before evaluation
        mcp_url = os.getenv("MCP_URL", "http://localhost:8000")
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{mcp_url}/stats/reset", timeout=5)
        except Exception:
            pass  # MCP stats are optional

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Starting evaluation of {len(sampled_queries)} queries...")
        )

        all_ndcg = []
        all_mrr = []
        all_precision = []
        all_recall = []
        
        errors = []
        successful_queries = 0
        start_time = time.time()

        for idx, (query_id, query_text) in enumerate(sampled_queries):
            if idx > 0 and idx % 10 == 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Progress: {idx}/{len(sampled_queries)} queries evaluated")
                )

            query_obj = RetrievalQuery(query=query_text, top_k=top_k)
            query_msg = query_obj.model_dump_json()

            try:
                response_text = await self.messenger.talk_to_agent(
                    message=query_msg,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=120
                )

                try:
                    response_data = json.loads(response_text)
                    response = RetrievalResponse.model_validate(response_data)
                    retrieved_doc_ids = response.doc_ids
                except (json.JSONDecodeError, ValidationError) as e:
                    errors.append({
                        "query_id": query_id,
                        "error": f"Invalid response format: {str(e)[:100]}"
                    })
                    continue

                query_qrels = self.qrels.get(query_id, {})
                total_relevant = sum(1 for score in query_qrels.values() if score > 0)
                
                relevances = []
                for doc_id in retrieved_doc_ids:
                    relevances.append(query_qrels.get(doc_id, 0))
                
                all_ndcg.append(compute_ndcg(relevances, top_k))
                all_mrr.append(compute_mrr(relevances, top_k))
                all_precision.append(compute_precision(relevances, top_k))
                all_recall.append(compute_recall(relevances, top_k, total_relevant))
                
                successful_queries += 1

            except Exception as e:
                errors.append({
                    "query_id": query_id,
                    "error": str(e)
                })

        execution_time = time.time() - start_time

        if successful_queries == 0:
            await updater.reject(new_agent_text_message("No valid responses received from purple agent"))
            return

        metrics = {}
        if all_ndcg:
            metrics[f"ndcg@{top_k}"] = round(np.mean(all_ndcg), 4)
        if all_mrr:
            metrics[f"mrr@{top_k}"] = round(np.mean(all_mrr), 4)
        if all_precision:
            metrics[f"precision@{top_k}"] = round(np.mean(all_precision), 4)
        if all_recall:
            metrics[f"recall@{top_k}"] = round(np.mean(all_recall), 4)

        # Fetch MCP server stats (tool call count)
        tool_calls = None
        try:
            async with httpx.AsyncClient() as client:
                stats_response = await client.get(f"{mcp_url}/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    tool_calls = stats.get("search_call_count")
        except Exception:
            pass  # MCP stats are optional

        results = {
            "assessment_type": "nfcorpus_retrieval",
            "num_queries": num_queries,
            "evaluated_queries": successful_queries,
            "failed_queries": len(errors),
            "top_k": top_k,
            "metrics": metrics,
            "tool_calls": tool_calls,
            "execution_time_seconds": round(execution_time, 2)
        }

        summary_text = f"""NFCorpus Retrieval Evaluation Results

Queries Evaluated: {successful_queries}/{num_queries}
Top-K: {top_k}

Metrics:
"""
        for metric_name, value in metrics.items():
            summary_text += f"  {metric_name}: {value:.4f}\n"
        
        summary_text += f"\nExecution Time: {execution_time:.2f} seconds\n"
        if tool_calls is not None:
            summary_text += f"Tool Calls (search_nfcorpus): {tool_calls}\n"
        if errors:
            summary_text += f"Failed Queries: {len(errors)}\n"

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=results))
            ],
            name="Evaluation Results",
        )
