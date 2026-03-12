import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置 =================
DEEPSEEK_API_KEY = "sk-422f9778baed40149bb65d5b008e8083"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

INPUT_FOLDER = "Human_json"
OUTPUT_FOLDER = "Human_json_cleaned_v2"
LOG_FOLDER = "processing_logs_v2"

# 创建输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# ================= LLM 提取模块 =================

def extract_title_with_llm(original_text):
    """
    从原始文本中提取论文题目（仅当原始文本包含完整题目时）
    不能搜索网络，只能从原始文本中提取
    """
    if not original_text or len(original_text.strip()) < 5:
        return None
    
    prompt = f"""
你是一个严格的数据提取助手。请从以下引用文本中提取论文题目。

原始文本：
{original_text}

规则：
1. 如果原始文本包含完整的论文题目，提取题目（不包含作者、年份、期刊等）
2. 如果原始文本只有期刊信息（如 "Phys. Rev. Lett. 78, 985 (1997)"），返回 null
3. 不能搜索网络，不能生成题目，只能提取原始文本中实际存在的题目
4. 题目通常在引号内、作者名之后、期刊名之前

请以 JSON 格式返回：
{{
    "has_title": true/false,
    "title": "提取的题目" 或 null
}}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的数据提取助手，只输出 JSON 格式。不能搜索网络，只能从提供的文本中提取。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        if result.get('has_title') and result.get('title'):
            return result['title'].strip()
    except Exception as e:
        pass
    return None

def extract_metadata_with_llm(original_text):
    """
    从原始文本中提取期刊元数据（仅当原始文本只有期刊信息时）
    不能搜索网络，只能从原始文本中提取
    """
    if not original_text or len(original_text.strip()) < 5:
        return None
    
    prompt = f"""
你是一个严格的数据提取助手。请从以下引用文本中提取期刊元数据。

原始文本：
{original_text}

规则：
1. 提取 ONLY: 'journal' (Journal Name), 'volume' (Volume Number), 'page' (Page Number/Article ID), 'year' (Year)
2. 如果字段缺失，使用 null
3. 不能搜索网络，不能生成信息，只能提取原始文本中实际存在的信息

请以 JSON 格式返回：
{{
    "journal": "期刊名" 或 null,
    "volume": "卷号" 或 null,
    "page": "页码" 或 null,
    "year": "年份" 或 null
}}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的数据提取助手，只输出 JSON 格式。不能搜索网络，只能从提供的文本中提取。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        metadata = json.loads(content)
        if metadata.get('journal') or (metadata.get('volume') and metadata.get('page')):
            return metadata
    except Exception as e:
        pass
    return None

def determine_format_type(original_text):
    """
    判断引用格式类型：是否有标题，是否是物理学格式
    返回: {
        'has_title': bool,
        'is_physics_format': bool,
        'should_delete': bool
    }
    """
    if not original_text or len(original_text.strip()) < 5:
        return {'has_title': False, 'is_physics_format': False, 'should_delete': True}
    
    prompt = f"""
你是一个严格的数据分析助手。请分析以下引用文本的格式类型。

原始文本：
{original_text}

分析规则：
1. **检查是否包含论文标题**：
   - 标题通常在引号内（如 "Title of the Paper"）
   - 或在作者名之后、期刊名之前
   - 有明显的标题格式（首字母大写、冒号后等）
   
2. **检查是否是物理学/APS格式**：
   - 如果文本明显是物理学引用（如 "Phys. Rev. Lett. 78, 985 (1997)"）
   - 且**不包含**论文标题
   - 则认为是物理学格式
   
3. **检查是否应该删除**：
   - 如果文本严重截断、乱码、或完全无效 → 应该删除

请以 JSON 格式返回：
{{
    "has_title": true/false,  // 原始文本是否包含论文标题
    "is_physics_format": true/false,  // 是否是物理学/APS格式（无标题，只有期刊信息）
    "should_delete": true/false  // 是否应该删除（无效/截断）
}}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的数据分析助手，只输出 JSON 格式。不能搜索网络，只能基于提供的文本进行分析。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return {
            'has_title': result.get('has_title', False),
            'is_physics_format': result.get('is_physics_format', False),
            'should_delete': result.get('should_delete', False)
        }
    except Exception as e:
        # 如果分析失败，默认不删除
        return {'has_title': False, 'is_physics_format': False, 'should_delete': False}

def process_single_reference(original_text, ref_id):
    """
    处理单条引用
    返回: {
        'title': str or None,
        'action': 'cleaned' | 'deleted',
        'reason': str
    }
    """
    # 1. 判断格式类型
    format_info = determine_format_type(original_text)
    
    if format_info['should_delete']:
        return {
            'title': None,
            'action': 'deleted',
            'reason': '原始文本无效或严重截断'
        }
    
    # 2. 根据格式类型处理
    if format_info['has_title']:
        # 原始文本有标题，提取标题
        extracted_title = extract_title_with_llm(original_text)
        if extracted_title:
            return {
                'title': extracted_title,
                'action': 'cleaned',
                'reason': '从原始文本中提取论文题目'
            }
        else:
            # 提取失败，尝试用期刊格式
            metadata = extract_metadata_with_llm(original_text)
            if metadata:
                journal = metadata.get('journal', '')
                volume = metadata.get('volume', '')
                page = metadata.get('page', '')
                year = metadata.get('year', '')
                
                # 清洗数据
                if volume: volume = str(volume).replace("Vol.", "").replace("vol.", "").strip()
                if page: page = str(page).replace("pp.", "").replace("Page", "").replace("page", "").strip()
                
                # 构造期刊格式
                parts = []
                if journal and journal.lower() != 'null' and journal.strip():
                    parts.append(journal)
                if volume and volume.lower() != 'null' and volume.strip():
                    parts.append(f"Vol. {volume}")
                if page and page.lower() != 'null' and page.strip():
                    parts.append(f"Page {page}")
                if year and year.lower() != 'null' and year.strip():
                    parts.append(f"({year})")
                
                constructed_title = ", ".join(parts)
                if len(constructed_title) >= 5:
                    return {
                        'title': constructed_title,
                        'action': 'cleaned',
                        'reason': '标题提取失败，使用期刊格式'
                    }
    
    # 3. 如果是物理学格式或没有标题，使用期刊格式
    if format_info['is_physics_format'] or not format_info['has_title']:
        metadata = extract_metadata_with_llm(original_text)
        if metadata:
            journal = metadata.get('journal', '')
            volume = metadata.get('volume', '')
            page = metadata.get('page', '')
            year = metadata.get('year', '')
            
            # 清洗数据
            if volume: volume = str(volume).replace("Vol.", "").replace("vol.", "").strip()
            if page: page = str(page).replace("pp.", "").replace("Page", "").replace("page", "").strip()
            
            # 构造期刊格式
            parts = []
            if journal and journal.lower() != 'null' and journal.strip():
                parts.append(journal)
            if volume and volume.lower() != 'null' and volume.strip():
                parts.append(f"Vol. {volume}")
            if page and page.lower() != 'null' and page.strip():
                parts.append(f"Page {page}")
            if year and year.lower() != 'null' and year.strip():
                parts.append(f"({year})")
            
            constructed_title = ", ".join(parts)
            if len(constructed_title) >= 5:
                return {
                    'title': constructed_title,
                    'action': 'cleaned',
                    'reason': '从原始文本中提取期刊信息并格式化为期刊格式'
                }
    
    # 4. 如果都失败了，删除
    return {
        'title': None,
        'action': 'deleted',
        'reason': '无法提取有效信息'
    }

def process_batch_references(references, batch_size=15):
    """
    批量处理引用（提高效率）
    """
    results = []
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, ref in enumerate(references):
            original_text = ref.get('text', '')
            future = executor.submit(process_single_reference, original_text, idx)
            futures[future] = idx
        
        # 收集结果（保持顺序）
        results_dict = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理引用", unit="ref"):
            ref_id = futures[future]
            try:
                result = future.result()
                results_dict[ref_id] = result
            except Exception as e:
                results_dict[ref_id] = {
                    'title': None,
                    'action': 'deleted',
                    'reason': f'处理失败: {str(e)}'
                }
            time.sleep(0.05)  # 避免API速率限制
        
        # 按原始顺序返回
        for idx in range(len(references)):
            results.append(results_dict.get(idx, {
                'title': None,
                'action': 'deleted',
                'reason': '处理失败'
            }))
    
    return results

def process_file(input_path, output_path, log_path):
    """
    处理单个文件
    """
    print(f"📄 正在处理: {os.path.basename(input_path)}", flush=True)
    
    # 读取原始文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    references = data.get('references', [])
    if not references:
        print(f"  ⚠️  文件无引用数据，但仍会保存到输出文件夹", flush=True)
        # 即使没有引用，也保存文件
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 生成日志文件
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 清洗日志: {os.path.basename(input_path)}\n\n")
            f.write(f"总引用数: 0\n")
            f.write(f"清洗后引用数: 0\n")
            f.write(f"删除引用数: 0\n\n")
            f.write("=" * 80 + "\n\n")
            f.write("⚠️  此文件无引用数据\n\n")
        
        print(f"  💾 已保存: {output_path}", flush=True)
        print(f"  📝 日志已保存: {log_path}\n", flush=True)
        return
    
    print(f"  📊 总引用数: {len(references)}", flush=True)
    
    # 批量处理引用
    results = process_batch_references(references)
    
    # 构建清洗后的数据
    cleaned_references = []
    log_entries = []
    
    for idx, (ref, result) in enumerate(zip(references, results)):
        if result['action'] == 'deleted':
            log_entries.append({
                'ref_id': idx,
                'original_text': ref.get('text', ''),
                'action': 'deleted',
                'reason': result['reason']
            })
        else:
            cleaned_ref = ref.copy()
            cleaned_ref['title'] = result['title']
            cleaned_references.append(cleaned_ref)
            
            log_entries.append({
                'ref_id': idx,
                'original_text': ref.get('text', ''),
                'cleaned_title': result['title'],
                'action': result['action'],
                'reason': result['reason']
            })
    
    # 更新数据
    data['references'] = cleaned_references
    
    # 保存清洗后的文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 生成日志文件
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"# 清洗日志: {os.path.basename(input_path)}\n\n")
        f.write(f"总引用数: {len(references)}\n")
        f.write(f"清洗后引用数: {len(cleaned_references)}\n")
        f.write(f"删除引用数: {len(references) - len(cleaned_references)}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write(f"## 引用 #{entry['ref_id']}\n\n")
            f.write(f"**原始文本**:\n```\n{entry['original_text']}\n```\n\n")
            
            if entry['action'] == 'deleted':
                f.write(f"❌ **已删除**\n\n")
                f.write(f"**删除原因**: {entry['reason']}\n\n")
            else:
                f.write(f"**清洗后标题**:\n```\n{entry['cleaned_title']}\n```\n\n")
                f.write(f"**处理说明**: {entry['reason']}\n\n")
            
            f.write("---\n\n")
    
    # 统计信息
    deleted_count = len(references) - len(cleaned_references)
    retention_rate = (len(cleaned_references) / len(references) * 100) if references else 0
    
    print(f"  ✅ 完成: 保留 {len(cleaned_references)} 条，删除 {deleted_count} 条 (保留率: {retention_rate:.1f}%)", flush=True)
    print(f"  💾 已保存: {output_path}", flush=True)
    print(f"  📝 日志已保存: {log_path}\n", flush=True)

def main():
    print("=" * 80, flush=True)
    print("🔧 开始清洗数据（基于排他性规则）", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    # 扫描所有文件
    print("📂 正在扫描文件...", flush=True)
    file_pairs = []
    
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith('.json'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, INPUT_FOLDER)
                subdir = os.path.dirname(rel_path)
                
                # 构建输出路径
                if subdir:
                    output_path = os.path.join(OUTPUT_FOLDER, subdir, file)
                    log_path = os.path.join(LOG_FOLDER, subdir, file.replace('.json', '_processing_log.md'))
                else:
                    output_path = os.path.join(OUTPUT_FOLDER, file)
                    log_path = os.path.join(LOG_FOLDER, file.replace('.json', '_processing_log.md'))
                
                # 检查是否已处理
                if os.path.exists(output_path):
                    print(f"⏭️  跳过已处理: {file}", flush=True)
                    continue
                
                file_pairs.append((input_path, output_path, log_path))
    
    if not file_pairs:
        print("✅ 所有文件已处理完成", flush=True)
        return
    
    print(f"✅ 找到 {len(file_pairs)} 个文件需要处理\n", flush=True)
    
    # 处理每个文件
    for idx, (input_path, output_path, log_path) in enumerate(file_pairs, 1):
        print(f"[{idx}/{len(file_pairs)}] ", end="", flush=True)
        try:
            process_file(input_path, output_path, log_path)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}\n", flush=True)
    
    print("=" * 80, flush=True)
    print("✅ 清洗完成！", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

