import logging
import os
import time
import json
import threading
import sqlite3
from typing import Any, Optional, Dict
from uuid import uuid4
import concurrent.futures
import yaml

from openai import OpenAI
from web_search_agent.web_search_agent import WebSearchAgent
from web_search_agent.search.search_api import web_search
from webpage import SearchResultInfo
from agent_action import ActionInfo
from context_store import append_action_info
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class WebSearchTool(BaseTool):
    """A tool for web searching using the web search agent."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        self.agent_config = self._load_agent_config()
        self.client = self._create_openai_client()
        self.web_search_agent = WebSearchAgent(config=self.agent_config, client=self.client)

        self._instance_dict = {}
        self.api_result_dict = {}
        self.id_to_context_lock = threading.Lock()
        self.instance_dict_lock = threading.Lock()
        # init SQLite cache
        self.query_db_path = self.agent_config["query_cache_db"]
        self._init_db(self.query_db_path)

        #  # Load cached search results
        # query_save_path = self.agent_config["query_save_path"]
        # query_save_path_dir = os.path.dirname(query_save_path)
        # if not os.path.exists(query_save_path_dir):
        #     os.makedirs(query_save_path_dir)
            
        # if os.path.exists(query_save_path):
        #     with open(query_save_path, 'r', encoding='utf-8') as f:
        #         self.api_result_dict = json.load(f)
        logger.info(f"WebSearchTool initialized with SQLite cache: {self.query_db_path}")

    def _load_agent_config(self) -> dict:
        config_path = "./user/tools/config.yaml"
        if os.path.exists(config_path):
            return yaml.safe_load(open(config_path))
        else:
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

    def _create_openai_client(self) -> OpenAI:
        """Create OpenAI client with default settings."""
        return OpenAI(
            api_key=self.agent_config.get("openai_api_key"),
            base_url=self.agent_config.get("openai_api_base", "https://openrouter.ai/api/v1")
        )

    def _init_db(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT PRIMARY KEY,
                timestamp REAL,
                organic TEXT
            )
        """)
        self.conn.commit()
        self.db_lock = threading.Lock()

    def _get_from_cache(self, query: str, ttl: int = 60*60*24*30) -> Optional[dict]:
        try:
            """Return cached result if still valid."""
            cur = self.conn.cursor()
            cur.execute("SELECT timestamp, organic FROM search_cache WHERE query = ?", (query,))
            row = cur.fetchone()
            logger.info(f"Cache lookup for query '{query}': {'HIT' if row else 'MISS'}")
            if row:
                ts, organic_str = row
                if time.time() - ts <= ttl:
                    return {"timestamp": ts, "organic": json.loads(organic_str)}
        except Exception as e:
            logger.error(f"Error getting cache for query '{query}': {str(e)}")
        return None

    def _set_cache(self, query: str, organic: list):
        """Insert or update cache."""
        try:
            with self.db_lock:  # 防止多线程写冲突
                self.conn.execute(
                    "INSERT OR REPLACE INTO search_cache (query, timestamp, organic) VALUES (?, ?, ?)",
                    (query, time.time(), json.dumps(organic, ensure_ascii=False))
                )
                self.conn.commit()
        except Exception as e:
            logger.error(f"Error setting cache for query '{query}': {str(e)}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, question: Optional[str] = None, **kwargs) -> str:
        with self.instance_dict_lock:
            if instance_id is None:
                if kwargs.get("create_kwargs", {}).get("request_id", {}):
                    logger.info(f"Using provided request_id as instance_id: {kwargs.get('create_kwargs', {}).get('request_id', {})}")
                    instance_id = kwargs.get("create_kwargs", {}).get("request_id", {})
                else:
                    logger.info("No instance_id provided, generating a new one.")
                    instance_id = str(uuid4())
            else:
                logger.info(f"Using provided instance_id: {instance_id}")
            if question is None:
                question = kwargs.get("create_kwargs", {}).get("question", "")
            self._instance_dict[instance_id] = {
                "question": question,
                "search_results": [],
                "context": []
            }
            return instance_id

    def search_and_add_to_cache(self, search_query: str, lock: threading.Lock, cur_api_result_dict):
        try:
            organic = web_search(search_query, self.agent_config)
            with lock:
                cur_api_result_dict[search_query]["organic"] = organic
                self._set_cache(search_query, organic)
        except Exception as e:
            logger.error(f"Error in search_and_add_to_cache for query '{search_query}': {str(e)}")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        start_time = time.perf_counter()
        logger.info(f"Executing web search for instance {instance_id} with parameters: {parameters}")
        # return [{"request_id": instance_id}], instance_id, {"search_results": [{"request_id": instance_id}]}
        try:
            search_query_list = parameters.get("query", [])
            if not isinstance(search_query_list, list):
                return "Error: query must be a list", 0, {}

            search_query_list = search_query_list[:3]
            question = self._instance_dict[instance_id]["question"]
            if not question:
                logger.error(f"Web Search Instance {instance_id} has no question set.")

            cur_api_result_dict = {}
            cur_api_result_dict_lock = threading.Lock()
            cache_hit = 0
            total_search_call = len(search_query_list)
            logger.info(f"Web search queries for instance {instance_id}: length {len(search_query_list)}")


            queries_to_search = []

            for query in search_query_list:
                cache_data = self._get_from_cache(query)
                if not isinstance(query, str):
                    continue
                if cache_data and len(cache_data["organic"]) > 0:
                    cache_hit += 1
                    self.api_result_dict[query] = cache_data
                    continue
                else:
                    queries_to_search.append(query)
                cur_api_result_dict[query] = {
                    "timestamp": time.time(),
                    "organic": []
                }

            # for query in search_query_list:
            #     if not isinstance(query, str):
            #         continue
            #     if (query in self.api_result_dict and 
            #         len(self.api_result_dict[query]['organic']) > 0 and 
            #         (time.time() - self.api_result_dict[query]['timestamp'] <= 60 * 60 * 24 * 7)):
            #         cache_hit += 1
            #         continue
            #     cur_api_result_dict[query] = {
            #         "timestamp": time.time(),
            #         "organic": []
            #     }
            
            # logger.info(f"Total search calls: {total_search_call}, Cache hits: {cache_hit}")
            logger.info(f"Total queries {len(search_query_list)}, cache hit {cache_hit}, need fresh {len(queries_to_search)}")

            # # 并发调用 API
            future_to_query = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as api_executor:
                # future_to_query = {executor.submit(self.search_and_add_to_cache, q): q for q in queries_to_search}
                for query in queries_to_search:
                    api_future = api_executor.submit(
                        self.search_and_add_to_cache,
                        query,
                        cur_api_result_dict_lock,
                        cur_api_result_dict
                    )
                    future_to_query.append(api_future)

            for api_future in concurrent.futures.as_completed(future_to_query):
                try:
                    api_future.result()
                except Exception as e:
                    logger.error(f"Error fetching search results for query: {str(e)}")
            # # 非并发调用 API
            # for q in queries_to_search:
            #     self.search_and_add_to_cache(q)

            # 收集结果
            # for query in search_query_list:
            #     cache_data = self._get_from_cache(query)
            #     if cache_data and len(cache_data["organic"]) > 0:
            #         logger.info(f"Using cached data for query '{query}'")
            #     else:
            #         logger.error(f"No data found for query '{query}' after search")
            #         cache_data = {"timestamp": time.time(), "organic": []}
            #     self.api_result_dict[query] = cache_data
            for k, v in cur_api_result_dict.items():
                if k not in self.api_result_dict:
                    self.api_result_dict[k] = v

            # 调用 web search agent 组织结果
            web_page_info_list_batch = self.web_search_agent.search_web_batch(
                user_query=question, 
                search_query_list=search_query_list, 
                api_result_dict=self.api_result_dict
            )

            search_result_info_list = [
                SearchResultInfo(
                    search_query=search_query_list[j],
                    web_page_info_list=web_page_info_list
                ) for j, web_page_info_list in enumerate(web_page_info_list_batch)
            ]

            cur_action_info = ActionInfo(
                user_query=question,
                search_thinking="",
                search_query_list=search_query_list,
                search_result_info_list=search_result_info_list
            )
            action_info_dict = {
                "user_query": cur_action_info.user_query,
                "search_query_list": cur_action_info.search_query_list,
                "search_result_info_list": [
                    {
                        "search_query": sr.search_query,
                        "web_page_info_list": [
                            {
                                "title": wp.title,
                                "url": wp.url,
                                "quick_summary": wp.quick_summary,
                                "browser": None,
                                "sub_question": sr.search_query
                            } for wp in sr.web_page_info_list
                        ]
                    } for sr in cur_action_info.search_result_info_list
                ]
            }
            append_action_info(instance_id, action_info_dict)

            self._instance_dict[instance_id]["context"].append(cur_action_info)

            content = []
            for search_result_info in search_result_info_list:
                search_query = search_result_info.search_query
                ret_web_page_info_list = []
                for web_page_info in search_result_info.web_page_info_list:
                    ret_web_page_info_list.append({
                        "title": web_page_info.title,
                        "url": web_page_info.url,
                        "quick_summary": web_page_info.quick_summary,
                        "browser": None,
                        "sub_question": search_query
                    })
                content.append({
                    "search_query": search_query,
                    "web_page_info_list": ret_web_page_info_list
                })
            content.append({"request_id": instance_id})

            self._instance_dict[instance_id]["search_results"] = content
            logger.info(f"Web search completed for instance {instance_id}, total results: {len(content)}")
            return json.dumps(content, ensure_ascii=False), instance_id, {"search_results": content}

        except Exception as e:
            logger.error(f"Error in web search execution for {instance_id}: {str(e)}")
            return f"Error: {str(e)}", 0, {}
        finally:
            elapsed = time.perf_counter() - start_time
            logger.warning(f"Execution web search time for instance {instance_id}: {elapsed:.4f}s")

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
