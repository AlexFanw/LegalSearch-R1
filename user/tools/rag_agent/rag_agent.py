import logging
import os
import time
import json
import threading
import concurrent.futures
import re
from typing import List, Dict, Any, Optional, Union

# RAG 依赖
import numpy as np
import pandas as pd
import faiss
import pickle
import jieba
from text2vec import SentenceModel
from rank_bm25 import BM25Okapi
from openai import OpenAI
from user.tools.utils import get_response_from_llm
from datetime import datetime

# 假设的辅助工具，用于 LLM 抽取
# from utils import get_response_from_llm # 这里我们直接在 Agent 中实现 LLM 调用

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# --- 全局RAG组件加载 (模拟服务启动时加载) ---
# 注意：在实际的 Agent 框架中，这些全局变量可能需要封装到 Agent 实例中或使用单例模式。

PROMPT_FOR_EXTRACT = """
请你从给定的中文查询语句（query）中抽取以下结构化信息，并以标准JSON格式输出。
不要输出任何解释或额外文字，只返回 JSON。

---

### 1. 时间信息（`time_info`）

* 输出格式：`YYYY-MM-DD`
* 若只出现年份或年月，需展开为起止日期：
  * 2024年 → ["2024-01-01", "2024-12-31"]
  * 2018年6月 → ["2018-06-01", "2018-06-30"]
* 若出现多个时间段，全部提取：
  * "2023年12月至2024年3月" → ["2023-12-01","2024-03-01"]
* 无时间信息则返回空列表 []。

---

### 2. 章节条目（chapter_info）

* 章节形式如“第X条”“第X章”等。
* 必须是 **全中文数字**（如“第三条”“第一百一十五条”）。
* 阿拉伯数字需转换为中文数字：

  * "刑法第3条和第10条" → ["第三条","第十条"]
* 无章节信息则返回空列表 []。

---

### 3. 关键词（keywords）

* 抽取主要法律、概念类关键词。
* 必须是语义完整的词。
* 不能是未出现在查询中的词。
* 若包含组合词，也需包含其子词：

  * "故意杀人罪" → 同时包含 "故意杀人"。

---

### 输出格式

```json
{
  "time_info": [],
  "chapter_info": [],
  "keywords": []
}
```

---

### 示例

**示例 1**

**输入：**
`2024年刑法中对故意杀人罪的量刑是什么？`

**输出：**

```json
{
  "time_info": ["2024-01-01", "2024-12-31"],
  "chapter_info": [],
  "keywords": ["故意杀人罪", "故意杀人", "杀人", "量刑"]
}
```

---

**示例 2**

**输入：**
2024年刑法第3条是什么？

**输出：**

```json
{
  "time_info": ["2024-01-01", "2024-12-31"],
  "chapter_info": ["第三条"],
  "keywords": ["刑法"]
}
```

---

**示例 3**

**输入：**
2019年义务教育法相关法律条文 2006年修订版

**输出：**

```json
{
  "time_info": ["2019-01-01", "2019-12-31", "2006-01-01", "2006-12-31"],
  "chapter_info": [],
  "keywords": ["义务教育法"]
}
```

"""

class RagGlobalComponent:
    """RAG 核心组件的单例加载和存储"""
    _instance = None
    _lock = threading.Lock()
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RagGlobalComponent, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any]):
        if self._is_initialized:
            return

        self.config = config
        self.global_model = None
        self.index = None
        self.metadata_df = None
        self.original_texts = []
        self.tokenized_texts_for_bm25 = []
        self.bm25 = None
        self.openai_client = None
        self._load_components()
        self._is_initialized = True

    def _load_components(self):
        """加载所有 RAG 依赖组件：模型、索引、元数据、BM25"""
        logger.info("正在加载文本嵌入模型...")
        self.global_model = SentenceModel(self.config["embedding_model_path"])
        logger.info("文本嵌入模型加载完成。")
        
        save_dir = self.config["rag_data_dir"]
        index_path = f"{save_dir}/faiss_index.bin"
        metadata_df_path = f"{save_dir}/metadata_df.pkl"
        original_texts_path = self.config["original_texts_path"]

        logger.info("正在加载 FAISS 索引...")
        self.index = faiss.read_index(index_path)
        logger.info(f"FAISS 索引加载完成，包含 {self.index.ntotal} 个向量。")

        logger.info("正在加载 Metadata...")
        with open(metadata_df_path, 'rb') as f:
            self.metadata_df = pickle.load(f)

        logger.info("正在加载 Original Texts...")
        with open(original_texts_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    title = data.get("title", "")
                    if title:
                        concatenated_text = (
                            f"路径：{data['path'].replace('> 未分编部分 ', '')} "
                            f"内容：{'【' + title + '】' + data['text'].replace(data['article_no'], '')}"
                        )
                    else:
                        concatenated_text = (
                            f"路径：{data['path'].replace('> 未分编部分 ', '')} "
                            f"内容：{data['text'].replace(data['article_no'], '')}"
                        )
    
                    self.original_texts.append(concatenated_text.replace("> 未分节部分 ", ""))

        # --- 初始化 BM25 ---
        logger.info("正在初始化 BM25...")
        self.tokenized_texts_for_bm25 = [list(jieba.cut(text)) for text in self.original_texts]

        self.bm25 = BM25Okapi(
            corpus=self.tokenized_texts_for_bm25,
            k1=self.config.get("bm25_k1", 1.2),
            b=self.config.get("bm25_b", 0.75),
            epsilon=self.config.get("bm25_epsilon", 0.25)
        )
        logger.info("BM25 初始化完成。")
        
        # --- 初始化 OpenAI 客户端 ---
        self.openai_client = OpenAI(
            base_url=self.config["llm_base_url"],
            api_key=self.config["llm_api_key"],
        )


class RagAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 初始化全局组件，确保只加载一次
        self.components = RagGlobalComponent(config)
        self.llm_model = config.get("llm_extract_model", "qwen/qwen3-30b-a3b-instruct-2507")
        self.llm_temp = config.get("llm_extract_temp", 0.1)

    # --- 辅助方法：JSON提取与时间处理 ---

    def _extract_json_from_response(self, response_text: str) -> str:
        """从API响应中提取JSON部分"""
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        if matches:
            return matches[0]
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            return response_text[start:end]
        raise ValueError("无法从响应中提取有效的JSON")

    def _process_extracted_info(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """处理提取的信息，将时间字符串转换为 datetime 对象"""
        processed_result = {
            "time_info": [],
            "chapter_info": raw_result.get("chapter_info", []),
            "keywords": raw_result.get("keywords", [])
        }
        
        time_info = raw_result.get("time_info", [])
        if time_info:
            processed_time_info = []
            for time_str in time_info:
                try:
                    dt = datetime.strptime(time_str, "%Y-%m-%d")
                    processed_time_info.append(dt)
                except ValueError as e:
                    logger.error(f"时间格式解析错误: {time_str}, 错误: {e}")
            processed_result["time_info"] = processed_time_info
        
        return processed_result
        
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空的结果字典"""
        return {"time_info": [], "chapter_info": [], "keywords": []}
    
    # --- 核心方法：信息抽取 ---

    def _extract_and_parse_query_info(self, query: str) -> Dict[str, Any]:
        """
        从查询中抽取时间信息、章节条目信息、关键词等结构化信息 (使用 LLM)
        """
        # prompt_for_extract= """
        #     我刚给你了一个query,请你帮我从中抽取时间信息,章节条目信息（必须是全中文形式，没有阿拉伯数字）,关键词等结构化信息,请你整理成一个json格式的内容返回给我,
        #     以下是2个示例
        #     第一个示例的query是[2024年刑法中对故意杀人罪的量刑是什么？]
        #     返回内容是{
        #         "time_info": ["2024-01-01", "2024-12-31"],
        #         "chapter_info":[],
        #         "keywords": ["故意杀人罪", "故意杀人","杀人","量刑"] 
        #     } 
        #     第二个示例的query是[2024年刑法第3条是什么？]
        #     其返回json是{
        #         "time_info": ["2024-01-01", "2024-12-31"],
        #         "chapter_info":["第三条"],
        #         "keywords": ["刑法"] 
        #     } 
        #     第三个示例的query是[2019年义务教育法相关法律条文 2006年修订版]
        #     其返回json是{
        #         "time_info": ["2019-01-01", "2019-12-31"],
        #         "chapter_info":[],
        #         "keywords": ["义务教育法"] 
        #     }
        #     1.注意对time_info时间的抽取要求如下:
        #     返回的年月日,格式必须是YYYY-MM-DD格式，
        #     如果只抽取到年/年月,比如2024年,2018年6月，应该返回该时间对应的头尾日期，比如2024年对应["2024-01-01", "2024-12-31"],2018年6月对应["2018-06-01","2018-06-30"]，
        #     如果query中包含多个时间,请全部抽取出来并返回,比如"2023年12月至2024年3月期间的刑法修正案"对应的time_info是["2023-12-01","2024-03-01"],比如“2018年刑法第四条和2024年刑法第四条的区别是什么”对应的time_info是["2018-01-01","2018-12-31","2024-01-01","2024-12-31"]，
        #     如果query中没有时间信息,time_info请返回空列表[]，
        #     2.注意对chapter_info章节条目的抽取要求如下:
        #     章节条目chapter_info一定是全中文的形式，没有阿拉伯数字，
        #     章节条目一般以"第X条"、"第X章"等形式出现,请全部抽取出来并返回,
        #     如果是阿拉伯数字3，101等需要转换为中文数字"三"，"一百零一"等形式返回，
        #     例如"刑法第3条和第10条的区别是什么"对应的chapter_info是["第三条","第十条"]，
        #     例如“民事诉讼法第一百一十五条”对应的chapter_info是["第一百一十五条"]，
        #     如果query中没有章节条目信息,chapter_info请返回空列表[]，
        #     请确保返回的是有效的JSON格式，不要包含任何解释文本。
        #     3.注意关键词抽取要求如下:
        #     抽取出来的词必须是完整的词，
        #     如果抽取出来的词内还有语义成立的词也要抽取,比如抽取出了“故意杀人罪”,还应该有“故意杀人”。
        #     """
        prompt_for_extract = PROMPT_FOR_EXTRACT
        max_retries_config = self.config.get("llm_extract_retries", 3)
        try:
            max_retries = int(max_retries_config)
        except (TypeError, ValueError):
            logger.warning("无效的 llm_extract_retries 配置值 %s，使用默认值 3。", max_retries_config)
            max_retries = 3
        max_retries = max(1, max_retries)

        retry_delay_config = self.config.get("llm_extract_retry_delay", 2.0)
        try:
            retry_delay = float(retry_delay_config)
        except (TypeError, ValueError):
            logger.warning("无效的 llm_extract_retry_delay 配置值 %s，使用默认值 2.0。", retry_delay_config)
            retry_delay = 2.0

        backoff_multiplier_config = self.config.get("llm_extract_retry_backoff", 1.0)
        try:
            backoff_multiplier = float(backoff_multiplier_config)
        except (TypeError, ValueError):
            logger.warning("无效的 llm_extract_retry_backoff 配置值 %s，使用默认值 1.0。", backoff_multiplier_config)
            backoff_multiplier = 1.0

        last_exception: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                response = get_response_from_llm(
                    messages=[
                        {"role": "system", "content": "你是一个法律智能助手。请严格按照要求返回JSON格式，不要添加任何解释文本。"},
                        {"role": "user", "content": f"query内容: {query}\n\n" + prompt_for_extract},
                    ],
                    client=self.components.openai_client,
                    model=self.llm_model,
                    stream=False,
                    temperature=self.llm_temp,
                    timeout=60,
                )

                response_text = (response or {}).get("content", "").strip()
                if not response_text:
                    raise ValueError("LLM信息抽取结果为空响应。")

                json_str = self._extract_json_from_response(response_text)
                raw_result = json.loads(json_str)
                processed_result = self._process_extracted_info(raw_result)

                return processed_result

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        "LLM信息抽取第 %d 次失败，将在 %.2f 秒后重试: %s",
                        attempt,
                        retry_delay,
                        e,
                    )
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                        retry_delay *= backoff_multiplier
                else:
                    logger.error("LLM信息抽取在最大重试次数后失败: %s", e)

        if last_exception:
            logger.error("LLM信息抽取最终失败，返回空结果。最后错误: %s", last_exception)
        return self._create_empty_result()


    # --- 核心方法：RAG检索 ---

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        RAG 检索主函数，执行过滤、混合检索和 RRF 重排。
        """
        start_time = time.perf_counter()
        logger.info(f"开始 RAG 检索: {query}")
        
        # 获取 RAG 组件
        model = self.components.global_model
        index = self.components.index
        metadata_df = self.components.metadata_df.copy() # 拷贝防止修改原始 df
        original_texts = self.components.original_texts
        bm25 = self.components.bm25
        tokenized_texts_for_bm25 = self.components.tokenized_texts_for_bm25

        # a. ===================================信息抽取
        result = self._extract_and_parse_query_info(query)
        print(f"抽取结果: {result}")
        dates: List[datetime] = result['time_info']
        chapter_info: List[str] = result['chapter_info']
        keywords: List[str] = result['keywords']
        logger.info(f"LLM抽取结果 - 时间: {dates}, 章节: {chapter_info}, 关键词: {keywords}")

        # b. ===================================时间过滤逻辑
        if not dates:
            filtered_ids = list(metadata_df['id'])
            logger.info("未检测到时间信息，使用全部 ID 进行检索。")
        else: 
            # 多个时间，取最小和最大形成区间
            start_dt = min(dates)
            end_dt = max(dates)
            # 找到在 [start_dt, end_dt] 区间内 *任意时刻* 有效的法条
            filtered_ids = metadata_df[
                (metadata_df['valid_from_dt'] <= end_dt) &
                (metadata_df['valid_to_dt'] >= start_dt)
            ]['id'].tolist()
            logger.info(f"时间区间 [{start_dt.date()}, {end_dt.date()}] 过滤后，候选 ID 数量: {len(filtered_ids)}")

        if not filtered_ids:
            logger.warning("在指定时间范围内未找到任何法条。")
            return []
        
        filtered_metadata_df = metadata_df[metadata_df['id'].isin(filtered_ids)]
        
        # c.=====================================章节条目过滤
        if chapter_info:
            chapter_pattern = '|'.join(chapter_info)
            print(f"章节条目过滤模式: {chapter_pattern}")
            filtered_metadata_df = filtered_metadata_df[
                filtered_metadata_df['article_no'].str.contains(chapter_pattern, na=False)
            ]
            filtered_ids = filtered_metadata_df['id'].tolist()
            logger.info(f"章节条目过滤后，候选 ID 数量: {len(filtered_ids)}")
            if not filtered_ids:
                logger.warning("在指定章节条目范围内未找到任何法条。")
                return []
        
        # --- 核心检索步骤只在 filtered_ids 范围内进行 ---
        
        # d.=======================================关键词精准匹配检索
        keyword_top_k_global_ids = []
        if keywords:
            def calculate_keyword_score(text, keywords):
                total_score = 0
                for keyword in keywords:
                    count = text.count(keyword)
                    if count > 0:
                        total_score += 1.0 + 0.1 * (count - 1)
                return total_score / len(keywords) if keywords else 0
            
            keyword_results = []
            for fid in filtered_ids:
                score = calculate_keyword_score(original_texts[fid], keywords)
                if score > 0:
                    keyword_results.append((fid, score))
            
            keyword_results.sort(key=lambda x: x[1], reverse=True)
            keyword_top_k_global_ids = [int(item[0]) for item in keyword_results[:k]]
            logger.info(f"关键词匹配 Top-{k} IDs: {keyword_top_k_global_ids}")
            
        # e. =====================================BM25 稀疏检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)
        filtered_bm25_scores = np.array([bm25_scores[i] for i in filtered_ids])
        
        # 避免空数组
        if filtered_bm25_scores.size == 0:
            bm25_top_k_global_ids = []
        else:
            top_k_bm25 = min(k, len(filtered_bm25_scores))
            bm25_top_k_indices_in_subset = np.argsort(filtered_bm25_scores)[::-1][:top_k_bm25]
            bm25_top_k_global_ids = [int(filtered_ids[i]) for i in bm25_top_k_indices_in_subset]
        logger.info(f"BM25 Top-{k} IDs: {bm25_top_k_global_ids}")


        # f. Dense 检索
        query_embedding = model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        all_k = min(index.ntotal, k * 10)
        distances, indices = index.search(query_embedding, all_k)

        dense_results = []
        for dist, idx in zip(distances[0], indices[0]):
            idx_int = int(idx)
            if idx_int in filtered_ids:
                dense_results.append((idx_int, float(dist)))
            if len(dense_results) >= k:
                break
        dense_top_k_global_ids = [int(item[0]) for item in dense_results]
        logger.info(f"Dense Top-{k} IDs: {dense_top_k_global_ids}")

        # g. RRF 重排 (Reciprocal Rank Fusion)
        
        def calculate_rrf_score(rank, k_value=60):
            return 1.0 / (k_value + rank)
        
        all_candidate_ids = {
            int(doc_id)
            for doc_id in (set(keyword_top_k_global_ids) | set(bm25_top_k_global_ids) | set(dense_top_k_global_ids))
        }
        
        # 设置不同检索方法的权重 (根据您的测试文件设置)
        keyword_weight = 3.0    
        dense_weight = 2.0      
        bm25_weight = 1.0       
        
        final_scores = {}
        
        for doc_id in all_candidate_ids:
            rrf_score = 0.0
            
            # 1. 关键词 RRF
            if doc_id in keyword_top_k_global_ids:
                keyword_rank = keyword_top_k_global_ids.index(doc_id) + 1
                rrf_score += keyword_weight * calculate_rrf_score(keyword_rank)
            
            # 2. Dense RRF
            if doc_id in dense_top_k_global_ids:
                dense_rank = dense_top_k_global_ids.index(doc_id) + 1
                rrf_score += dense_weight * calculate_rrf_score(dense_rank)
            
            # 3. BM25 RRF
            if doc_id in bm25_top_k_global_ids:
                bm25_rank = bm25_top_k_global_ids.index(doc_id) + 1
                rrf_score += bm25_weight * calculate_rrf_score(bm25_rank)
            
            final_scores[int(doc_id)] = rrf_score

        # 按 RRF 分数排序并获取最终 Top-k
        final_top_k_global_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
        
        # h. 构建返回结果
        results = []
        for pid in final_top_k_global_ids:
            # 确保从完整的 metadata_df 获取，因为 filtered_metadata_df 只是子集
            row = metadata_df[metadata_df['id'] == pid].iloc[0]
            score_value = float(final_scores.get(pid, 0.0))

            valid_from = row.get('valid_from') if hasattr(row, 'get') else row['valid_from']
            valid_to = row.get('valid_to') if hasattr(row, 'get') else row['valid_to']

            if pd.isna(valid_from):
                valid_from = None
            else:
                valid_from = str(valid_from)

            if pd.isna(valid_to):
                valid_to = None
            else:
                valid_to = str(valid_to)

            pid_int = int(pid)
            # results.append({
            #     "id": pid_int,
            #     "original_text": str(original_texts[pid_int]),
            #     "valid_from": valid_from,
            #     "valid_to": valid_to,
            #     "final_score": score_value  # 附带最终分数
            # })
            results.append({
                "id": pid_int,
                "法条内容": str(original_texts[pid_int]),
                "施行时间": valid_from,
                "失效时间": valid_to,
                "相关度分数": score_value  # 附带最终分数
            })

        elapsed = time.perf_counter() - start_time
        logger.info(f"RAG 检索完成，返回 {len(results)} 个结果，耗时 {elapsed:.4f}s")
        return results