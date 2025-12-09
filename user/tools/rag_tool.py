import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from context_store import append_action_info
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .rag_agent.rag_agent import RagAgent
from .schemas import OpenAIFunctionToolSchema


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RAGTool(BaseTool):
	"""Tool wrapper around the internal RagAgent for legal text retrieval."""

	def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
		super().__init__(config, tool_schema)

		self._config = self._load_rag_config()
		self._agent_lock = threading.Lock()
		self._instance_lock = threading.Lock()
		self._rag_agent: Optional[RagAgent] = None
		self._instance_dict: Dict[str, Dict[str, Any]] = {}

		self._max_queries = int(self._config.get("max_queries_per_call", 3))
		self._top_k = int(self._config.get("retrieval_top_k", 5))

		logger.info(
			"RAGTool initialized with model=%s, data_dir=%s",
			self._config.get("embedding_model_path"),
			self._config.get("rag_data_dir"),
		)

		try:
			self._get_agent()
			logger.info("RagAgent preloaded successfully during initialization")
		except Exception as exc:
			logger.error("Failed to preload RagAgent during initialization: %s", exc)
			raise

	# ------------------------------------------------------------------
	# Configuration helpers
	# ------------------------------------------------------------------
	def _load_rag_config(self) -> Dict[str, Any]:
		"""Load RagAgent configuration with sensible fallbacks."""

		config_path = Path("./user/tools/config.yaml")
		loaded_cfg: Dict[str, Any] = {}

		if config_path.exists():
			try:
				loaded_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
				logger.info("Loaded RAG config from %s", config_path)
			except Exception as exc:
				logger.warning("Failed to load RAG config %s: %s", config_path, exc)

		# defaults
		default_cfg = {
			"embedding_model_path": os.getenv(
				"RAG_EMBEDDING_MODEL", "GanymedeNil/text2vec-large-chinese"
			),
			"rag_data_dir": os.getenv(
				"RAG_DATA_DIR", "./user/tools/rag_agent/processed/vectors"
			),
			"original_texts_path": os.getenv(
				"RAG_ORIGINAL_TEXTS_PATH",
				"./user/tools/rag_agent/processed/chunks/law_chunks.jsonl",
			),
			"bm25_k1": float(os.getenv("RAG_BM25_K1", "1.2")),
			"bm25_b": float(os.getenv("RAG_BM25_B", "0.75")),
			"bm25_epsilon": float(os.getenv("RAG_BM25_EPSILON", "0.25")),
			"llm_base_url": os.getenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
			"llm_api_key": os.getenv("RAG_LLM_API_KEY"),
			"llm_extract_model": os.getenv(
				"RAG_LLM_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"
			),
			"llm_extract_temp": float(os.getenv("RAG_LLM_TEMP", "0.1")),
			# "retrieval_top_k": int(os.getenv("RAG_TOP_K", "5")),
			"max_queries_per_call": int(os.getenv("RAG_MAX_QUERIES", "3")),
		}

		combined_cfg = {**default_cfg, **loaded_cfg}

		if not combined_cfg.get("llm_api_key"):
			fallback_key = self._load_fallback_openai_key()
			if fallback_key:
				combined_cfg["llm_api_key"] = fallback_key

		if not combined_cfg.get("llm_api_key"):
			raise ValueError(
				"Missing llm_api_key for RAG tool. Set RAG_LLM_API_KEY env or provide in config."
			)

		return combined_cfg

	def _load_fallback_openai_key(self) -> Optional[str]:
		"""Attempt to reuse the OpenRouter key from the shared config file."""

		shared_config = Path("./user/tools/config.yaml")
		if not shared_config.exists():
			return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

		try:
			data = yaml.safe_load(shared_config.read_text(encoding="utf-8")) or {}
		except Exception as exc:
			logger.warning("Failed to parse shared config %s: %s", shared_config, exc)
			data = {}

		if isinstance(data, dict):
			for key_name in ("rag_llm_api_key", "openai_api_key", "openrouter_api_key"):
				if data.get(key_name):
					return data[key_name]

		return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

	# ------------------------------------------------------------------
	# Lifecycle helpers
	# ------------------------------------------------------------------
	def _get_agent(self) -> RagAgent:
		if self._rag_agent is None:
			with self._agent_lock:
				if self._rag_agent is None:
					logger.info("Creating RagAgent instance... this may take a while")
					self._rag_agent = RagAgent(self._config)
		return self._rag_agent

	def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
		return self.tool_schema

	async def create(
		self,
		instance_id: Optional[str] = None,
		query: Optional[str] = None,
		**kwargs,
	) -> str:
		with self._instance_lock:
			create_kwargs = kwargs.get("create_kwargs", {})
			if instance_id is None:
				instance_id = (
					create_kwargs.get("request_id")
					or kwargs.get("request_id")
					or str(uuid4())
				)
				logger.info("RAGTool generated new instance_id=%s", instance_id)
			else:
				logger.info("RAGTool reuse instance_id=%s", instance_id)

			query = query or create_kwargs.get("query") or create_kwargs.get("question") or ""

			self._instance_dict[instance_id] = {
				"base_query": query,
				"results": [],
				"last_run": None,
			}

			return instance_id

	@rollout_trace_op
	async def execute(
		self,
		instance_id: str,
		parameters: dict[str, Any],
		**kwargs,
	) -> tuple[str, float, dict]:
		start_time = time.perf_counter()
		logger.info("Executing RAG retrieval for instance %s", instance_id)

		try:
			if instance_id not in self._instance_dict:
				return "Error: instance not found", 0.0, {}

			query_list = parameters.get("query", [])
			if not isinstance(query_list, list) or not query_list:
				return "Error: query must be a non-empty list", 0.0, {}

			# Deduplicate while preserving order
			seen = set()
			unique_queries: List[str] = []
			for item in query_list:
				if not isinstance(item, str):
					continue
				normalized = item.strip()
				if normalized and normalized not in seen:
					unique_queries.append(normalized)
					seen.add(normalized)

			if not unique_queries:
				return "Error: query list contains no valid strings", 0.0, {}

			unique_queries = unique_queries[: self._max_queries]
			agent = self._get_agent()

			retrieval_records: List[Dict[str, Any]] = []
			total_hits = 0

			for single_query in unique_queries:
				logger.info("RAG retrieving for query='%s'", single_query)
				try:
					results = agent.retrieve(single_query, k=self._top_k)
				except Exception as exc:
					logger.error("RAG retrieval failed for '%s': %s", single_query, exc)
					results = []

				formatted_results: List[Dict[str, Any]] = []
				for item in results or []:
					formatted_results.append(
						{
							"id": item.get("id"),
							"法条内容": item.get("法条内容", ""),
							"施行时间": item.get("施行时间"),
							"失效时间": item.get("失效时间"),
							"相关度分数": item.get("相关度分数"),
						}
					)

				retrieval_records.append(
					{
						"query": single_query,
						"result_count": len(formatted_results),
						"results": formatted_results,
					}
				)
				total_hits += len(formatted_results)

			instance_state = self._instance_dict.get(instance_id)
			if instance_state is not None:
				instance_state["results"] = retrieval_records
				instance_state["last_run"] = time.time()

			append_action_info(
				instance_id,
				{
					"tool": "rag_retrieve",
					"rag_query_list": unique_queries,
					"rag_results": retrieval_records,
				},
			)

			response_payload = json.dumps(retrieval_records, ensure_ascii=False)
			metrics = {
				"rag_results": retrieval_records,
				"query_count": len(retrieval_records),
				"total_hits": total_hits,
			}

			logger.info(
				"RAG retrieval finished for instance %s in %.2fs (hits=%d)",
				instance_id,
				time.perf_counter() - start_time,
				total_hits,
			)

			return response_payload, 0.0, metrics

		except Exception as exc:
			logger.error("Unexpected error during RAG execution: %s", exc)
			return f"Error: {exc}", 0.0, {}

	async def release(self, instance_id: str, **kwargs) -> None:
		with self._instance_lock:
			if instance_id in self._instance_dict:
				del self._instance_dict[instance_id]
				logger.info("Released RAG instance %s", instance_id)
