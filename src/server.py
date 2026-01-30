import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()
    
    skill = AgentSkill(
        id="nfcorpus_retrieval_eval",
        name="NFCorpus Biomedical Retrieval Evaluation",
        description="Evaluates an agent's ability to retrieve relevant biomedical documents from the NFCorpus corpus. The agent receives biomedical queries and must return ranked lists of relevant document IDs. Performance is measured using standard IR metrics: NDCG@k, MRR@k, Precision@k, and Recall@k.",
        tags=["evaluation", "information-retrieval", "biomedical", "nfcorpus", "beir"],
        examples=["""
{
  "participants": {
    "retrieval_agent": "https://retrieval.example.com:9010"
  },
  "config": {
    "num_queries": 50,
    "top_k": 10,
    "random_seed": 777
  }
}
"""]
    )

    agent_card = AgentCard(
        name="NFCorpus Retrieval Evaluator",
        description="Green agent that evaluates purple agents on the NFCorpus biomedical information retrieval benchmark. Sends biomedical queries to purple agents, collects ranked document IDs, and reports comprehensive IR metrics including NDCG, MRR, Precision, and Recall.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
