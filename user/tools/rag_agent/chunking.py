import re
import json
from pathlib import Path
import os 
import pandas as pd
from datetime import datetime

orginal_txt_dir = "/Users/wengyanbing/Independent_project/2025Fall/LAW RAG/RAG_pipeline/corpus/original_txt"
menu_file_path = "/Users/wengyanbing/Independent_project/2025Fall/LAW RAG/RAG_pipeline/corpus/menu/all_law_menu.csv"

# 读取menu_file_path对应的csv文件，获取menu_df
menu_df = pd.read_csv(menu_file_path)
print(menu_df.head(5))

# law_type = '中华人民共和国刑法'
# law_name = '中华人民共和国刑法(2023修正)'
output_dir_path = '/Users/wengyanbing/Independent_project/2025Fall/LAW RAG/RAG_pipeline/processed/chunks/'

valid_from = datetime.strptime('2023-12-31', '%Y-%m-%d')
valid_to = datetime.strptime('2099-12-31', '%Y-%m-%d')

# 检查output_dir是否存在
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)


def chunk_law_text(text: str, valid_from: datetime.day = valid_from, valid_to: datetime.day = valid_to,law_title: str = "中华人民共和国刑法（2017修正）"):
    """
    从原始法律文本中提取结构化层级：
    编 -> 章 -> 节 -> 条
    """

    # Step 1: 清洗文本（保留换行，用于层级识别）
    text = text.replace('\r', '')
    text = re.sub(r'　', ' ', text)  # 去掉全角空格
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text.strip())

    # Step 2: 找到正文起点
    # 常见正文起始标志：第一编 / 第一条
    start_match = re.search(r'(^|\n)(第[一二三四五六七八九十百千万零〇两\d]+(编|条))', text)
    if start_match:
        text = text[start_match.start():]  # 从正文处截断
    else:
        print("未检测到明确正文起点，保留全文")


    # Step 2: 定义层级模式（注意中文数字及“之”）
    part_pat = re.compile(r'^第[一二三四五六七八九十百千万零〇两\d]+编', re.M)
    chapter_pat = re.compile(r'^第[一二三四五六七八九十百千万零〇两\d]+章', re.M)
    section_pat = re.compile(r'^第[一二三四五六七八九十百千万零〇两\d]+节', re.M)
    article_pat = re.compile(r'^第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十百千万零〇两\d]+)?', re.M)

    # Step 3: 初始化层级容器
    parts = []
    current_part = None
    current_chapter = None
    current_section = None

    # Step 4: 按行解析
    lines = text.split('\n')
    buffer = ""  # 累积当前条文正文

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # --- 匹配“编” ---
        if part_pat.match(line):
            current_part = {"name": line.strip(), "chapters": []}
            # print(f"Detected part: {current_part['name']}")
            parts.append(current_part)
            current_chapter = None
            current_section = None
            continue

        # --- 匹配“章” ---
        if chapter_pat.match(line):
            current_chapter = {"name": line.strip(), "sections": []}
            # print(f"Detected chapter: {current_chapter['name']}")
            if current_part is None:
                current_part = {"name": "未分编部分", "chapters": []}
                # print(f"Auto-created part: {current_part['name']}")
                parts.append(current_part)
            current_part["chapters"].append(current_chapter)
            # print(f"Appended chapter to part: {current_part['name']}")
            current_section = None
            continue

        # --- 匹配“节” ---
        if section_pat.match(line):
            current_section = {"name": line.strip(), "articles": []}
            if current_chapter is None:
                current_chapter = {"name": "未分章部分", "sections": []}
                current_part["chapters"].append(current_chapter)
            current_chapter["sections"].append(current_section)
            continue

        # --- 匹配“条” ---
        if article_pat.match(line):
            # 如果上一个条正文还在 buffer 中，先写入
            if buffer and current_section and len(current_section["articles"]) > 0:
                current_section["articles"][-1]["text"] += " " + buffer.strip()
                buffer = ""

            article_no = article_pat.match(line).group(0)
            title_match = re.search(r'【(.*?)】', line)
            title = title_match.group(1) if title_match else None
            content = re.sub(r'^第.*?[】）]\s*', '', line).strip()

            article = {
                "index": article_no,
                "title": title,
                "text": content
            }

            # 挂载条文
            if current_section:
                current_section["articles"].append(article)
            elif current_chapter:
                current_chapter["sections"].append({
                    "name": "未分节部分",
                    "articles": [article]
                })
                current_section = current_chapter["sections"][-1]
            elif current_part:
                current_part["chapters"].append({
                    "name": "未分章部分",
                    "sections": [{
                        "name": "未分节部分",
                        "articles": [article]
                    }]
                })
                current_section = current_part["chapters"][-1]["sections"][-1]
            else:
                # 没有任何层级，自动新建
                current_part = {"name": "未分编部分", "chapters": [{
                    "name": "未分章部分",
                    "sections": [{
                        "name": "未分节部分",
                        "articles": [article]
                    }]
                }]}
                parts.append(current_part)
                current_section = current_part["chapters"][0]["sections"][0]
            continue

        # --- 如果不是层级标识，就视为正文内容 ---
        buffer += " " + line

    # Step 5: 收尾，把最后一条正文补上
    if buffer and current_section and len(current_section["articles"]) > 0:
        current_section["articles"][-1]["text"] += " " + buffer.strip()

    return {
        "title": law_title,
        "valid_from": valid_from.strftime('%Y-%m-%d') if isinstance(valid_from, datetime) else str(valid_from),
        "valid_to": valid_to.strftime('%Y-%m-%d') if isinstance(valid_to, datetime) else str(valid_to),
        "parts": parts
    }

def flatten_articles(law_json):
    """从层级结构中提取所有条文，生成平面列表"""
    results = []
    law_title = law_json.get("title", "未知法律")
    valid_from = law_json.get("valid_from")
    valid_to = law_json.get("valid_to")

    for part in law_json.get("parts", []):
        part_name = part["name"]
        for chapter in part.get("chapters", []):
            chapter_name = chapter["name"]
            for section in chapter.get("sections", []):
                section_name = section["name"]
                for article in section.get("articles", []):
                    path = f"{law_title} > {part_name} > {chapter_name} > {section_name} > {article['index']}"
                    results.append({
                        "valid_from": valid_from,
                        "valid_to": valid_to,
                        "path": path,
                        "article_no": article["index"],
                        "title": article.get("title"),
                        "text": article.get("text", "").strip()
                    })
    return results

def save_to_jsonl(data, output_path="law_chunks.jsonl"):
    """保存为 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存到 {output_path}，共 {len(data)} 条。")

def extract_law_name_from_filename(filename):
    """
    从文件名中提取法律名称，保留有意义的括号内容，去除技术标识符
    
    规则：
    1. 去除 .txt 后缀
    2. 保留修正、修订等法律版本信息的括号
    3. 去除包含 FBM-CLI、数字编号等技术标识的括号
    
    Examples:
        "中华人民共和国刑法(FBM-CLI.1.556).txt" -> "中华人民共和国刑法"
        "中华人民共和国刑法(2009第二次修正)(FBM-CLI.1.329703).txt" -> "中华人民共和国刑法(2009第二次修正)"
        "民法典(2020年版)(FBM-CLI.2.123).txt" -> "民法典(2020年版)"
    """
    # 去除文件扩展名
    name = filename.replace('.txt', '').strip()
    
    # 方法1：使用正则表达式直接匹配和提取
    # 匹配模式：法律名称 + 可选的修正信息 + FBM-CLI技术标识
    pattern = r'^(.*?)(\(\d{4}.*?修正\))?\(FBM-CLI.*?\)$'
    match = re.match(pattern, name)
    
    if match:
        base_name = match.group(1).strip()  # 基础法律名称
        revision_info = match.group(2) if match.group(2) else ""  # 修正信息
        law_name = base_name + revision_info
        return law_name
    else:
        # 如果不匹配预期模式，使用备用方法
        # 去除所有包含FBM-CLI的括号
        law_name = re.sub(r'\(.*?FBM-CLI.*?\)', '', name).strip()
        return law_name
    
if __name__ == "__main__":
    # 从文件夹orginal_txt_dir中逐个读取法律法条的txt文件，进行chunking处理，并保存结果
    # orginal_txt_dir下包含子文件夹和txt文件，子文件夹中也包含txt文件，我希望能够递归读取所有txt文件
    # 然后对每个txt文件，到menu_df中根据law_name和law_type_version进行对应，获取该法律的valid_from和过期日期，作为chunking函数的参数
    # 对每个txt文件进行chunking，将所有文件chunking之后的chunks合并，一起保存到output_dir_path下的law_chunks.jsonl文件中

    txt_file_path_law_names = []
    for root, dirs, files in os.walk(orginal_txt_dir):
        for file in files:
            if file.endswith('.txt'):
                law_name = extract_law_name_from_filename(file)
                print(f"Processing file: {file}, extracted law_name: {law_name}")
                txt_file_path_law_names.append((os.path.join(root, file), law_name))
    print(f"找到 {len(txt_file_path_law_names)} 个txt文件进行处理。")
    # print(txt_file_path_law_names)

    all_law_chunks = []
          
    for txt_file_path_law_name in txt_file_path_law_names:
        path = txt_file_path_law_name[0]
        law_name = txt_file_path_law_name[1]
        print(f"\n处理文件: {path}")
        text = Path(path).read_text(encoding="utf-8")
        # 根据law_name到menu_df中查找对应的valid_from和valid_to
        matching_row = menu_df[menu_df['law_type_version'] == law_name]
        if not matching_row.empty:
            valid_from = matching_row.iloc[0]['valid_from']
            valid_to = matching_row.iloc[0]['valid_to']
            print(f"找到匹配记录:")
            print(f"valid_from: {valid_from}")
            print(f"valid_to: {valid_to}")
        else:
            print(f"未找到匹配记录，使用默认日期:")
            valid_from = '2023-12-31'
            valid_to = '2099-12-31'

        # =============================执行chunking
        data = chunk_law_text(text, valid_from=valid_from, valid_to=valid_to, law_title=law_name)
        print(f"成功分割 {len(data)} 条法条。")
        # print(json.dumps(data, ensure_ascii=False, indent=2))
        all_law_chunks.extend(flatten_articles(data))
    # =============================保存结果
    # 提取平面条文列表
    # flattened = flatten_articles(data)

    print(f"共提取 {len(all_law_chunks)} 条法条片段")
    # 保存为 JSONL
    output_path = os.path.join(output_dir_path, "law_chunks.jsonl")
    save_to_jsonl(all_law_chunks, output_path)
    print(f"已保存到 {output_path}")
    # 保存完整层级结构为 JSON
    with open(os.path.join(output_dir_path, "law_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)