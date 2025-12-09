#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检索功能测试
"""

import sys
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def test_simple_retrieve():
    """测试简单检索功能"""
    try:
        print("⏳ 初始化RagAgent...")
        start_time = time.time()
        from rag_agent.rag_agent import RagAgent
        
        config = {
            'embedding_model_path': 'GanymedeNil/text2vec-large-chinese',
            'rag_data_dir': './rag_agent/processed/vectors',
            'original_texts_path': './rag_agent/processed/chunks/law_chunks.jsonl',
            'bm25_k1': 1.2,
            'bm25_b': 0.75,
            'bm25_epsilon': 0.25,
            'llm_base_url': 'https://openrouter.ai/api/v1',
            'llm_api_key': 'sk-or-v1-xxx',  # 请替换为你的OpenAI API Key
            'llm_extract_model': 'qwen/qwen3-30b-a3b-instruct-2507',
            'llm_extract_temp': 0.1
        }
        
        agent = RagAgent(config)
        print("✅ RagAgent初始化完成")
        
        # 测试简单查询
        test_queries = [
            "中华人民共和国刑法第一百二十八条",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 测试查询 {i}: {query}")
            
            try:
                
                results = agent.retrieve(query, k=3)
                query_time = time.time() - start_time
                
                print(f"✅ 检索成功! 耗时: {query_time:.2f}秒")
                print(f"📊 返回结果数: {len(results)}")
                
                for j, result in enumerate(results):
                    print(result)
                    
            except Exception as e:
                print(f"❌ 检索失败: {e}")
                # 继续下一个查询
                continue
        
        print("\n🎉 快速检索测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_retrieve()
    sys.exit(0 if success else 1)
