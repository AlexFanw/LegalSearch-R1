import json
import numpy as np
import pandas as pd
from datetime import datetime
from text2vec import SentenceModel
import faiss
import re
import os
import pickle

# --- 1. 加载模型 ---
print("开始加载文本嵌入模型...")
# from text2vec import SentenceModel
model = SentenceModel("GanymedeNil/text2vec-large-chinese")
print("模型加载完成。")

def create_concatenated_text(data_item):
    """将条文内容和元数据拼接用于 embedding"""
    return (
        f"路径: {data_item['path']} | "
        f"条目: {data_item['article_no']} | "
        f"标题: {data_item['title']} | "
        f"内容: {data_item['text']}"
    )

# --- 2. 读取 JSONL 数据并预处理 ---
input_file = "./processed/chunks/law_chunks.jsonl"
texts = []
metadata_list = []
ids = []
original_texts = []

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            data = json.loads(line)
            concatenated_text = create_concatenated_text(data)
            original_texts.append(concatenated_text)  # 存储原始文本内容
            texts.append(concatenated_text)
            # 存储元数据，包含时间范围
            metadata_list.append({
                "id": i, # 使用行号作为 ID
                "valid_from": data["valid_from"],
                "valid_to": data["valid_to"],
                "path": data["path"],
                "article_no": data["article_no"],
                "title": data["title"],
            })
            ids.append(i) # ID 列表

print(f"共读取到 {len(texts)} 个文本块。")


# --- 3. 生成 Embeddings ---

print("正在生成 embeddings...")
# 尝试一个合适的批量大小，例如 32 或 64。如果内存不足，就减小它。
batch_size = 64 
# show_progress_bar=True 可以让您看到进度，确认它没有卡死
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
# embeddings = model.encode(texts)
dimension = embeddings.shape[1] 
print(f"Embeddings 形状: {embeddings.shape}")

# --- 4. 构建 FAISS 索引 ---
# 使用 IndexFlatIP (内积) 需要先对向量进行 L2 归一化
# 使用 IndexFlatL2 (L2 距离) 则不需要
index = faiss.IndexFlatIP(dimension) # 或 faiss.IndexFlatL2(dimension)
# 如果使用内积作为相似度，需要归一化
faiss.normalize_L2(embeddings)
index.add(embeddings.astype('float32'))

# --- 5. 将元数据转换为 Pandas DataFrame 便于查询 ---
metadata_df = pd.DataFrame(metadata_list)
# 将时间字符串转换为 datetime 对象，便于比较
metadata_df['valid_from_dt'] = pd.to_datetime(metadata_df['valid_from'])
metadata_df['valid_to_dt'] = pd.to_datetime(metadata_df['valid_to'])

print("FAISS 索引和元数据 DataFrame 构建完成。")


# 创建
save_dir = "./processed/vetors" # 建议创建一个专门的目录存放这些文件
os.makedirs(save_dir, exist_ok=True) # 创建目录（如果不存在）

# --- 8. 保存组件 ---
# a. 保存 FAISS 索引
index_path = os.path.join(save_dir, "faiss_index.bin")
faiss.write_index(index, index_path)
print(f"FAISS 索引已保存至: {index_path}")

# b. 保存 Embeddings (可选，通常 FAISS 索引已包含向量)
# 如果需要单独保存，例如用于其他分析或加载到不同索引类型
embeddings_path = os.path.join(save_dir, "embeddings.npy")
np.save(embeddings_path, embeddings)
print(f"Embeddings 已保存至: {embeddings_path}")

# c. 保存 Metadata DataFrame (使用 pickle)
metadata_df_path = os.path.join(save_dir, "metadata_df.pkl")
with open(metadata_df_path, 'wb') as f:
    pickle.dump(metadata_df, f)
print(f"Metadata DataFrame 已保存至: {metadata_df_path}")

# # d. 保存 Original Texts (使用 pickle)
# original_texts_path = os.path.join(save_dir, "original_texts.pkl")
# with open(original_texts_path, 'wb') as f:
#     pickle.dump(original_texts, f)
# print(f"Original Texts 已保存至: {original_texts_path}")

print("\n所有组件已成功保存到目录:", save_dir)
print("文件列表:")
for file in os.listdir(save_dir):
    print(f"  - {file}")