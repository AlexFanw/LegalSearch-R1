from typing import List, Dict, Any
from utils import (
    get_content_from_tag,
    get_response_from_llm
)
from .prompts import *
from webpage import *
import time
import random
import html2text
import concurrent.futures
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
import time

class ReadingAgent:
    def __init__(self,
                 config,
                 client):
        self.config = config
        self.client = client
        
    def read(
            self,
            main_question,
            sub_question,
            selected_result_idx: int,
            cur_webpage: WebPageInfo,
            context: List[WebPageInfo] | None = None,
            web_search_agent = None
    ):
        if context is None:
            context = []
        if cur_webpage.browser == "error":
            return cur_webpage
        if cur_webpage.browser is None:
            cur_webpage.browser = web_search_agent.scrape_and_check_valid_api(cur_webpage.url)
            if cur_webpage.browser is None:
                cur_webpage.browser = "error"
                return cur_webpage
        context_so_far_prefix = ""
        for webpage in context:
            useful_info = ""
            for page_read_info in webpage.page_read_info_list:
                useful_info += page_read_info.page_summary + "\n\n"
            if len(useful_info):
                context_so_far_prefix += f"<sub_question>{webpage.sub_question}</sub_question>\n<useful_info>{useful_info}</useful_info>\n"

        cur_useful_info = ""
        total_pages = len(cur_webpage.browser.viewport_pages)
        logger.info(f"Start reading webpage: {cur_webpage.url}, total pages: {total_pages}")
        if total_pages > 5:
            total_pages = 5
            logger.info(f"Limit to read first 5 pages of the webpage: {cur_webpage.url}")
        global_page_index = 0


        max_loops = total_pages + 2
        loop_cnt = 0
        
        while cur_webpage.browser.viewport_current_page < total_pages:
            loop_cnt += 1
            start_time = time.perf_counter()
            global_page_index += 1
            context_so_far = ""
            if cur_useful_info:
                context_so_far = context_so_far_prefix + f"<sub_question>{sub_question}</sub_question>\n<useful_info>{cur_useful_info}</useful_info>"
            else:
                context_so_far = context_so_far_prefix
            start_browse_time = time.perf_counter()
            cur_web_page_content = cur_webpage.browser._state()[1]
            cur_web_page_content = html2text.html2text(cur_web_page_content)
            end_browse_time = time.perf_counter()
            browse_elapsed_time = end_browse_time - start_browse_time
            logger.info(f"Webpage fetch time for page {cur_webpage.browser.viewport_current_page + 1} of {cur_webpage.url}: {browse_elapsed_time:.2f} seconds")
            # print("cur_web_page_content length: ", len(cur_web_page_content))
            # with open("cur_web_page_content.txt", "w", encoding="utf-8") as f:
            #     f.write("total_pages:\n")
            #     f.write(str(total_pages) + "\n")
            #     f.write("cur_web_page_content:\n")
            #     f.write(cur_web_page_content + "\n")
            page_index = cur_webpage.browser.viewport_current_page + 1
            prompt = EXTRACT_NEW_INFO_PROMPT.format(
                main_question=main_question,
                sub_question=sub_question,
                context_so_far=context_so_far.strip(),
                page_index=page_index,
                total_pages=total_pages,
                page_content=cur_web_page_content
            )

            messages = [{"role": "user", "content": prompt}]
            start_llm_time = time.perf_counter()

            response = get_response_from_llm(
                messages=messages,
                client=self.client,
                model=self.config["reading_agent_model"],
                stream=False,
                temperature=0.6,
                timeout=60,
            )
            # logger.info(f"LLM Response for page {page_index} of {cur_webpage.url}: {response['content']}")
            end_llm_time = time.perf_counter()
            llm_elapsed_time = end_llm_time - start_llm_time
            logger.info(f"LLM response time for page {page_index} of {cur_webpage.url}: {llm_elapsed_time:.2f} seconds")
            # with open("response.txt", "w
            
            extracted_info = get_content_from_tag(response["content"], "extracted_info", "").strip()
            page_down = get_content_from_tag(response["content"], "page_down", "").strip()
            short_summary = get_content_from_tag(response["content"], "short_summary", "").strip()

            if "yes" in page_down:
                page_down = True
            else:
                page_down = False

            if extracted_info:
                cur_webpage.page_read_info_list.append(
                    PageReadInfo(
                        search_results_idx=selected_result_idx,
                        url=cur_webpage.url,
                        page_title=cur_webpage.title,
                        fetch_res=cur_web_page_content,
                        page_thinking=response["reasoning_content"] if "reasoning_content" in response else "",
                        page_summary=extracted_info,
                        page_number=cur_webpage.browser.viewport_current_page,
                        need_page_down=page_down,
                        used=False,
                    )
                )
                cur_useful_info += extracted_info + "\n\n"
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info(f"Browse time for page {page_index} of {cur_webpage.url}: {elapsed_time:.2f} seconds")

            if page_down:
                old_page = cur_webpage.browser.viewport_current_page
                cur_webpage.browser.page_down()
                new_page = cur_webpage.browser.viewport_current_page
                logger.info(f"page_down from {old_page} -> {new_page} for {cur_webpage.url}")

                if new_page == old_page:
                    logger.warning(
                        f"page_down did not change page index for {cur_webpage.url}, "
                        f"breaking to avoid infinite loop."
                    )
                    break
            else:
                break

            if loop_cnt > max_loops:
                logger.warning(
                    f"Loop count exceeded for {cur_webpage.url} (>{max_loops}), "
                    f"breaking to avoid infinite loop."
                )
                break

        logger.info(f"Finished reading webpage: {cur_webpage.url}, total pages read: {global_page_index}")
        return cur_webpage

    def read_batch(
            self,
            user_query: str,
            search_result_info_list: List[SearchResultInfo],
            url_list: List[str],
            web_search_agent = None,
    ):
        url_allowlist = set(url_list)
        scheduled_urls = set()
        future_to_content = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for search_result_info in search_result_info_list:
                search_query = search_result_info.search_query
                web_page_info_list = search_result_info.web_page_info_list
                for selected_result_idx, cur_webpage in enumerate(web_page_info_list):
                    if cur_webpage.url not in url_allowlist:
                        continue
                    if cur_webpage.url in scheduled_urls:
                        continue
                    scheduled_urls.add(cur_webpage.url)
                    future = executor.submit(self.read,
                                            user_query,
                                            search_query,
                                            selected_result_idx,
                                            cur_webpage,
                                            web_page_info_list,
                                            web_search_agent)
                    future_to_content.append(future)
        read_webpage_list = []
        for i, future in enumerate(future_to_content):
            cur_webpage: WebPageInfo = future.result()
            read_webpage_list.append(cur_webpage)
        return read_webpage_list

                
                
