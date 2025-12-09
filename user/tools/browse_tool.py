# browse_tool.py
import logging
import os
import json
import yaml
from typing import Any, Optional, List
from uuid import uuid4
from openai import OpenAI

from reading_agent.reading_agent import ReadingAgent
from webpage import WebPageInfo
from verl.utils.rollout_trace import rollout_trace_op
from web_search_agent.web_search_agent import WebSearchAgent
from context_store import load_latest_context
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
import os
from pathlib import Path
import time
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARNING"))


class BrowseWebpageTool(BaseTool):
    """A tool for browsing webpages using the reading agent."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        # Load configuration from file
        self.agent_config = self._load_agent_config()
        
        # Create OpenAI client
        self.client = self._create_openai_client()
        
        # Initialize reading agent
        self.reading_agent = ReadingAgent(config=self.agent_config, client=self.client)
        self.web_search_agent = WebSearchAgent(config=self.agent_config, client=self.client)
        self._instance_dict = {}

    def _load_agent_config(self) -> dict:
        """Load agent configuration from file."""
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

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema


    async def create(self, instance_id: Optional[str] = None, question: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            if kwargs.get("create_kwargs", {}).get("request_id", {}):
                instance_id = kwargs.get("create_kwargs", {}).get("request_id", {})
                logger.info(f"Browse Request ID: {instance_id}")
            else:
                instance_id = str(uuid4())
                logger.info("No provided instance_id, generating a new one: %s", instance_id)
        if question is None:
            question = kwargs.get("create_kwargs", {}).get("question", "")
        self._instance_dict[instance_id] = {
            "question": question,
            "browsed_pages": []
        }
        return instance_id

    import time
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        start_time = time.perf_counter()
        try:
            url_list = parameters.get("url_list", [])
            if not isinstance(url_list, list) or len(url_list) == 0:
                return "Error: url_list must be a non-empty list", 0, {}
            
            question = self._instance_dict[instance_id]["question"]
            
            if not question:
                logger.error(f"Browse Instance {instance_id} has no question set.")
            else:
                logger.info(f"Browsing webpages for instance {instance_id} with question: {question}")
            
            context_data = None
            for attempt in range(3):
                context_data = load_latest_context(instance_id)
                if context_data:
                    logger.info(f"Context loaded for instance {instance_id} on attempt {attempt + 1}/3")
                    break
                logger.warning(f"Context not found for instance {instance_id}, retrying ({attempt + 1}/3)...")
                time.sleep(10) 

            logger.info("load search action info for instance: %s", instance_id)

            if not context_data or context_data.get("user_query") != question:
                if not context_data:
                    logger.error("Error: No matching search context found. Please search first: %s, User query: %s", instance_id, question)
                elif context_data.get("user_query") != question:
                    logger.error("Error: Search context user query does not match the current question: %s", instance_id)
                return "Error: No matching search context found. Please search first: %s, User query: %s" % (instance_id, question), 0, {}


            from webpage import SearchResultInfo, WebPageInfo
            search_result_info_list = []
            for sr in context_data["search_result_info_list"]:
                web_page_info_list = [WebPageInfo(**wp) for wp in sr["web_page_info_list"]]
                search_result_info_list.append(SearchResultInfo(
                    search_query=sr["search_query"],
                    web_page_info_list=web_page_info_list
                ))

            read_webpage_list: List[WebPageInfo] = self.reading_agent.read_batch(
                user_query=question,
                search_result_info_list=search_result_info_list,
                url_list=url_list[0:5],  # Limit to first 3 URLs to control cost
                web_search_agent=self.web_search_agent
            )
            
            # Format response
            content = []
            for read_webpage in read_webpage_list:
                information = []
                for page_read_info in read_webpage.page_read_info_list:
                    if page_read_info.used:
                        continue
                    information.append({
                        "page_number": page_read_info.page_number,
                        "page_summary": page_read_info.page_summary
                    })
                    page_read_info.used = True
                
                content.append({
                    "url": read_webpage.url,
                    "information": information
                })
            
            self._instance_dict[instance_id]["browsed_pages"].extend(content)
            logger.info(f"Browsed {len(content)} pages for instance {instance_id}")
            # logger.warn(f"First 100 tokens of content: {json.dumps(content, ensure_ascii=False)[:100]}...")
            return json.dumps(content, ensure_ascii=False), 0.0, {"browsed_content": content}
            
        except Exception as e:
            logger.error(f"Error in browse webpage execution: {str(e)}")
            return f"Error: {str(e)}", 0, {}
        finally:
            end_time = time.perf_counter()  # ⏱️ 结束时间
            elapsed_time = end_time - start_time
            # print("page number: ", len(self._instance_dict[instance_id]["browsed_pages"])) if instance_id in self._instance_dict else None
            total_info_count = sum(
                len(page["information"]) for page in self._instance_dict[instance_id]["browsed_pages"]
            ) if instance_id in self._instance_dict else 0
            logger.warning(f"Total information pieces browsed for instance {instance_id}: {total_info_count}")
            # print("Execution browse time: %.2f seconds" % elapsed_time)
            logger.warning(f"Execution browse time for instance {instance_id}: {elapsed_time:.2f} seconds")
    # async def calc_reward(self, instance_id: str, **kwargs) -> float:
    #     # Reward based on the amount of content successfully browsed
    #     browsed_pages = self._instance_dict[instance_id]["browsed_pages"]
    #     total_info_count = sum(
    #         len(page["information"]) for page in browsed_pages
    #     )
    #     return min(total_info_count * 0.1, 1.0)  # Cap at 1.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]