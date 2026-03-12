#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查并修复脚本（使用cleaning_v2.py的逻辑）
- 发现问题的同时修复
- 检查删错、缺失、无中生有、标题丢失
"""

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
CLEANED_FOLDER = "Human_json_cleaned_v2"
LOG_FOLDER = "processing_logs_v2"

# ================= LLM 提取模块（与cleaning_v2.py一致）=================

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
    处理单条引用（与cleaning_v2.py逻辑一致）
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

# ================= 检查逻辑 =================

def check_and_fix_single_ref(ref_id, original_text, current_cleaned_title, current_action):
    """
    检查并修复单条引用
    返回: {
        'ref_id': int,
        'issues': list,  # 发现的问题
        'fixed': bool,   # 是否修复
        'new_title': str or None,
        'new_action': str,
        'new_reason': str
    }
    """
    result = {
        'ref_id': ref_id,
        'issues': [],
        'fixed': False,
        'new_title': current_cleaned_title,
        'new_action': current_action,
        'new_reason': ''
    }
    
    # 使用cleaning_v2.py的逻辑重新处理（带错误处理）
    try:
        correct_result = process_single_reference(original_text, ref_id)
        correct_title = correct_result.get('title')
        correct_action = correct_result.get('action')
        correct_reason = correct_result.get('reason')
    except Exception as e:
        # 如果处理失败，抛出异常让调用者重试
        raise Exception(f"处理引用 #{ref_id} 时出错: {str(e)}")
    
    # 检查问题
    issues = []
    
    # 1. 检查是否删错（不应该删除的删除了）
    if current_action == 'deleted' and correct_action == 'cleaned':
        issues.append({
            'type': '删错',
            'message': f"引用 #{ref_id} 被错误删除，应该保留",
            'original_text': original_text[:150],
            'correct_title': correct_title
        })
        result['fixed'] = True
        result['new_title'] = correct_title
        result['new_action'] = 'cleaned'
        result['new_reason'] = f"修复删错：{correct_reason}"
    
    # 2. 检查标题丢失（有标题但提取了期刊）
    elif current_action == 'cleaned' and correct_action == 'cleaned':
        # 检查当前标题是否是期刊格式
        is_journal_format = 'Vol.' in (current_cleaned_title or '') or 'Page' in (current_cleaned_title or '')
        # 检查正确结果是否是标题
        is_title_format = correct_title and not ('Vol.' in correct_title or 'Page' in correct_title)
        
        if is_journal_format and is_title_format:
            # 当前是期刊格式，但应该是标题
            issues.append({
                'type': '标题丢失',
                'message': f"引用 #{ref_id} 原始文本包含标题，但清洗后变成了期刊格式",
                'original_text': original_text[:150],
                'current_title': current_cleaned_title,
                'correct_title': correct_title
            })
            result['fixed'] = True
            result['new_title'] = correct_title
            result['new_action'] = 'cleaned'
            result['new_reason'] = f"修复标题丢失：{correct_reason}"
        
        # 检查标题是否无中生有（不在原始文本中）
        elif current_cleaned_title and correct_title:
            # 检查当前标题是否在原始文本中
            current_title_clean = re.sub(r'[^\w\s]', ' ', current_cleaned_title.lower())
            original_text_clean = re.sub(r'[^\w\s]', ' ', original_text.lower())
            
            # 检查标题的主要单词是否在原始文本中
            title_words = [w for w in current_title_clean.split() if len(w) > 2]
            if title_words:
                found_words = sum(1 for w in title_words[:5] if w in original_text_clean)
                if found_words < len(title_words[:5]) * 0.6:  # 少于60%的单词在原始文本中
                    issues.append({
                        'type': '无中生有',
                        'message': f"引用 #{ref_id} 清洗后的标题可能包含原始文本中不存在的内容",
                        'original_text': original_text[:150],
                        'current_title': current_cleaned_title,
                        'correct_title': correct_title
                    })
                    result['fixed'] = True
                    result['new_title'] = correct_title
                    result['new_action'] = 'cleaned'
                    result['new_reason'] = f"修复无中生有：{correct_reason}"
        
        # 检查标题是否正确
        elif current_cleaned_title != correct_title:
            issues.append({
                'type': '标题不正确',
                'message': f"引用 #{ref_id} 清洗后的标题不正确",
                'original_text': original_text[:150],
                'current_title': current_cleaned_title,
                'correct_title': correct_title
            })
            result['fixed'] = True
            result['new_title'] = correct_title
            result['new_action'] = 'cleaned'
            result['new_reason'] = f"修复标题：{correct_reason}"
    
    # 3. 检查是否应该删除但没删除
    elif current_action == 'cleaned' and correct_action == 'deleted':
        issues.append({
            'type': '应该删除',
            'message': f"引用 #{ref_id} 应该被删除但未删除",
            'original_text': original_text[:150],
            'current_title': current_cleaned_title
        })
        result['fixed'] = True
        result['new_title'] = None
        result['new_action'] = 'deleted'
        result['new_reason'] = correct_reason
    
    result['issues'] = issues
    return result

# ================= 文件处理 =================

def update_log_file(log_path, log_entries, original_refs):
    """更新日志文件"""
    if not os.path.exists(log_path):
        return
    
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    for ref_id, log_entry in log_entries.items():
        pattern = rf'## 引用 #{ref_id}\n\n\*\*原始文本\*\*:\n```\n(.*?)\n```\n\n(.*?)(?=\n---|\n## 引用 #|\Z)'
        match = re.search(pattern, log_content, re.DOTALL)
        if match:
            original_text_block = match.group(1)
            rest_content = match.group(2)
            
            if log_entry['action'] == 'deleted':
                # 更新为删除状态
                new_block = f"## 引用 #{ref_id}\n\n**原始文本**:\n```\n{original_text_block}\n```\n\n❌ **已删除**\n\n**删除原因**: {log_entry['reason']}\n"
                log_content = log_content[:match.start()] + new_block + log_content[match.end():]
            else:
                # 更新清洗后标题
                title_pattern = r'\*\*清洗后标题\*\*:\n```\n.*?\n```'
                new_title_block = f"**清洗后标题**:\n```\n{log_entry['cleaned_title']}\n```"
                
                if re.search(title_pattern, rest_content, re.DOTALL):
                    rest_content = re.sub(title_pattern, new_title_block, rest_content, flags=re.DOTALL)
                else:
                    if '❌ **已删除**' not in rest_content:
                        rest_content = new_title_block + '\n\n' + rest_content
                
                # 更新处理说明
                if log_entry.get('reason'):
                    reason_pattern = r'\*\*处理说明\*\*:.*?(?=\n|$)'
                    new_reason = f"**处理说明**: {log_entry['reason']}"
                    if re.search(reason_pattern, rest_content):
                        rest_content = re.sub(reason_pattern, new_reason, rest_content)
                    else:
                        if '❌ **已删除**' not in rest_content:
                            rest_content = rest_content.rstrip() + f'\n\n{new_reason}\n'
                
                new_block = f"## 引用 #{ref_id}\n\n**原始文本**:\n```\n{original_text_block}\n```\n\n{rest_content}"
                log_content = log_content[:match.start()] + new_block + log_content[match.end():]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

def check_and_fix_file(original_path, cleaned_path, log_path):
    """检查并修复单个文件"""
    file_name = os.path.basename(original_path)
    
    # 读取文件
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    original_refs = original_data.get('references', [])
    
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)
    cleaned_refs = cleaned_data.get('references', [])
    
    # 解析log文件
    log_entries = {}
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        pattern = r'## 引用 #(\d+)\n\n\*\*原始文本\*\*:\n```\n(.*?)\n```\n\n(.*?)(?=\n---|\n## 引用 #|\Z)'
        matches = re.findall(pattern, log_content, re.DOTALL)
        
        for match in matches:
            ref_id = int(match[0])
            original_text = match[1].strip()
            rest_content = match[2]
            
            entry = {
                'original_text': original_text,
                'cleaned_title': None,
                'action': None,
                'reason': ''
            }
            
            if '❌ **已删除**' in rest_content:
                entry['action'] = 'deleted'
                reason_match = re.search(r'\*\*删除原因\*\*: (.*?)(?=\n|$)', rest_content)
                if reason_match:
                    entry['reason'] = reason_match.group(1).strip()
            else:
                title_match = re.search(r'\*\*清洗后标题\*\*:\n```\n(.*?)\n```', rest_content, re.DOTALL)
                if title_match:
                    entry['cleaned_title'] = title_match.group(1).strip()
                entry['action'] = 'cleaned'
                reason_match = re.search(r'\*\*处理说明\*\*: (.*?)(?=\n|$)', rest_content)
                if reason_match:
                    entry['reason'] = reason_match.group(1).strip()
            
            log_entries[ref_id] = entry
    
    # 检查每个引用
    all_issues = []
    needs_update = False
    
    # 构建cleaned引用映射
    cleaned_idx = 0
    orig_to_cleaned = {}
    for orig_idx in range(len(original_refs)):
        if orig_idx in log_entries:
            log_e = log_entries[orig_idx]
            if log_e.get('action') != 'deleted':
                orig_to_cleaned[orig_idx] = cleaned_idx
                cleaned_idx += 1
    
    # 准备检查任务
    check_tasks = []
    for idx in range(len(original_refs)):
        orig_ref = original_refs[idx]
        original_text = orig_ref.get('text', '')
        log_entry = log_entries.get(idx, {})
        current_title = log_entry.get('cleaned_title', '')
        current_action = log_entry.get('action', 'cleaned')
        
        # 获取cleaned引用
        cleaned_ref = None
        if idx in orig_to_cleaned:
            cleaned_idx = orig_to_cleaned[idx]
            if cleaned_idx < len(cleaned_refs):
                cleaned_ref = cleaned_refs[cleaned_idx]
                if not current_title and cleaned_ref:
                    current_title = cleaned_ref.get('title', '')
        
        check_tasks.append((idx, original_text, current_title, current_action))
    
    # 并发检查（带重试机制）
    print(f"  正在检查 {len(check_tasks)} 个引用...", flush=True)
    check_results = []
    
    def check_with_retry(task, max_retries=3):
        """带重试的检查函数"""
        ref_id, original_text, current_title, current_action = task
        for attempt in range(max_retries):
            try:
                result = check_and_fix_single_ref(ref_id, original_text, current_title, current_action)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s, 6s
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次重试失败，返回默认结果（保持原状）
                    print(f"    ⚠️  引用 #{ref_id} 检查失败（已重试{max_retries}次）: {e}，将保持原状", flush=True)
                    return {
                        'ref_id': ref_id,
                        'issues': [],
                        'fixed': False,
                        'new_title': current_title,
                        'new_action': current_action,
                        'new_reason': log_entries.get(ref_id, {}).get('reason', '')
                    }
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_with_retry, task): task for task in check_tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  检查 {file_name[:40]}", unit="ref"):
            try:
                result = future.result()
                check_results.append(result)
            except Exception as e:
                # 如果连重试都失败了，创建默认结果保持原状
                task = futures[future]
                ref_id = task[0]
                original_text = task[1]
                current_title = task[2]
                current_action = task[3]
                print(f"    ⚠️  引用 #{ref_id} 检查异常: {e}，将保持原状", flush=True)
                check_results.append({
                    'ref_id': ref_id,
                    'issues': [],
                    'fixed': False,
                    'new_title': current_title,
                    'new_action': current_action,
                    'new_reason': log_entries.get(ref_id, {}).get('reason', '')
                })
    
    # 确保所有引用都有结果（防止遗漏）
    result_ref_ids = {r.get('ref_id') for r in check_results}
    for idx in range(len(original_refs)):
        if idx not in result_ref_ids:
            # 这个引用没有结果，创建默认结果保持原状
            orig_ref = original_refs[idx]
            original_text = orig_ref.get('text', '')
            log_entry = log_entries.get(idx, {})
            current_title = log_entry.get('cleaned_title', '')
            current_action = log_entry.get('action', 'cleaned')
            
            print(f"    ⚠️  引用 #{idx} 未检查（可能被遗漏），将保持原状", flush=True)
            check_results.append({
                'ref_id': idx,
                'issues': [],
                'fixed': False,
                'new_title': current_title,
                'new_action': current_action,
                'new_reason': log_entry.get('reason', '')
            })
    
    # 按ref_id排序
    check_results.sort(key=lambda x: x.get('ref_id', 0))
    
    # 应用修复（确保所有引用都被处理）
    new_cleaned_refs = []
    new_log_entries = {}
    processed_ref_ids = set()
    
    for result in check_results:
        ref_id = result['ref_id']
        if ref_id >= len(original_refs):
            continue  # 跳过无效的ref_id
        
        original_text = original_refs[ref_id].get('text', '')
        processed_ref_ids.add(ref_id)
        
        if result['fixed']:
            needs_update = True
            all_issues.extend(result['issues'])
            
            if result['new_action'] == 'deleted':
                # 不添加到cleaned_refs
                new_log_entries[ref_id] = {
                    'original_text': original_text,
                    'cleaned_title': None,
                    'action': 'deleted',
                    'reason': result['new_reason']
                }
            else:
                # 添加到cleaned_refs
                new_cleaned_refs.append({
                    'title': result['new_title'],
                    'text': original_text
                })
                new_log_entries[ref_id] = {
                    'original_text': original_text,
                    'cleaned_title': result['new_title'],
                    'action': 'cleaned',
                    'reason': result['new_reason']
                }
        else:
            # 保持原样
            if result['new_action'] == 'deleted':
                new_log_entries[ref_id] = {
                    'original_text': original_text,
                    'cleaned_title': None,
                    'action': 'deleted',
                    'reason': log_entries.get(ref_id, {}).get('reason', '')
                }
            else:
                if ref_id in orig_to_cleaned:
                    cleaned_idx = orig_to_cleaned[ref_id]
                    if cleaned_idx < len(cleaned_refs):
                        new_cleaned_refs.append(cleaned_refs[cleaned_idx])
                new_log_entries[ref_id] = log_entries.get(ref_id, {
                    'original_text': original_text,
                    'cleaned_title': result['new_title'],
                    'action': 'cleaned',
                    'reason': ''
                })
    
    # 确保所有原始引用都被处理（防止遗漏）
    for idx in range(len(original_refs)):
        if idx not in processed_ref_ids:
            # 这个引用没有被处理，保持原状
            original_text = original_refs[idx].get('text', '')
            log_entry = log_entries.get(idx, {})
            
            if log_entry.get('action') == 'deleted':
                new_log_entries[idx] = {
                    'original_text': original_text,
                    'cleaned_title': None,
                    'action': 'deleted',
                    'reason': log_entry.get('reason', '')
                }
            else:
                if idx in orig_to_cleaned:
                    cleaned_idx = orig_to_cleaned[idx]
                    if cleaned_idx < len(cleaned_refs):
                        new_cleaned_refs.append(cleaned_refs[cleaned_idx])
                new_log_entries[idx] = log_entry if log_entry else {
                    'original_text': original_text,
                    'cleaned_title': '',
                    'action': 'cleaned',
                    'reason': ''
                }
    
    # 检查缺失（cleaned文件中的引用数应该等于非删除的引用数）
    expected_count = sum(1 for e in new_log_entries.values() if e.get('action') != 'deleted')
    actual_count = len(new_cleaned_refs)
    
    if actual_count != expected_count:
        all_issues.append({
            'type': '缺失',
            'message': f"引用数量不匹配：期望 {expected_count} 个，实际 {actual_count} 个"
        })
        needs_update = True
    
    # 保存修复后的文件
    if needs_update:
        cleaned_data['references'] = new_cleaned_refs
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        if log_path and os.path.exists(log_path):
            update_log_file(log_path, new_log_entries, original_refs)
        
        print(f"  🔧 修复了 {len([i for i in all_issues if i['type'] != '缺失'])} 个问题", flush=True)
        if actual_count != expected_count:
            print(f"  ⚠️  引用数量不匹配：期望 {expected_count} 个，实际 {actual_count} 个", flush=True)
    
    return {
        'file': file_name,
        'total_original': len(original_refs),
        'total_cleaned': len(new_cleaned_refs),
        'issues': all_issues,
        'fixed_count': len([r for r in check_results if r['fixed']])
    }

def main():
    print("=" * 80, flush=True)
    print("🔍 检查并修复清洗结果（使用cleaning_v2.py逻辑）", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    # 扫描所有文件
    print("📂 正在扫描文件...", flush=True)
    file_pairs = []
    
    for root, dirs, files in os.walk(CLEANED_FOLDER):
        for file in files:
            if file.endswith('.json'):
                cleaned_path = os.path.join(root, file)
                rel_path = os.path.relpath(cleaned_path, CLEANED_FOLDER)
                subdir = os.path.dirname(rel_path)
                
                if subdir:
                    original_path = os.path.join(INPUT_FOLDER, subdir, file)
                    log_path = os.path.join(LOG_FOLDER, subdir, file.replace('.json', '_processing_log.md'))
                else:
                    original_path = os.path.join(INPUT_FOLDER, file)
                    log_path = os.path.join(LOG_FOLDER, file.replace('.json', '_processing_log.md'))
                
                if os.path.exists(original_path) and os.path.exists(cleaned_path):
                    file_pairs.append((original_path, cleaned_path, log_path))
    
    if not file_pairs:
        print("❌ 没有找到需要检查的文件对", flush=True)
        return
    
    print(f"✅ 找到 {len(file_pairs)} 个文件需要检查\n", flush=True)
    
    # 检查每个文件
    all_results = []
    total_issues = 0
    total_fixed = 0
    
    for idx, (original_path, cleaned_path, log_path) in enumerate(file_pairs, 1):
        file_name = os.path.basename(original_path)
        print(f"[{idx}/{len(file_pairs)}] 检查: {file_name}", flush=True)
        
        try:
            result = check_and_fix_file(original_path, cleaned_path, log_path)
            all_results.append(result)
            
            issue_count = len(result['issues'])
            fixed_count = result['fixed_count']
            
            total_issues += issue_count
            total_fixed += fixed_count
            
            if issue_count > 0:
                print(f"  ⚠️  发现 {issue_count} 个问题，修复了 {fixed_count} 个", flush=True)
            else:
                print(f"  ✅ 检查通过", flush=True)
            print(flush=True)
            
        except Exception as e:
            print(f"  ❌ 检查失败: {e}\n", flush=True)
            all_results.append({
                'file': file_name,
                'error': str(e)
            })
    
    # 生成报告
    print("=" * 80, flush=True)
    print("📊 检查总结", flush=True)
    print("=" * 80, flush=True)
    
    total_files = len(all_results)
    files_with_issues = sum(1 for r in all_results if r.get('issues'))
    files_ok = total_files - files_with_issues
    
    print(f"总文件数: {total_files}", flush=True)
    print(f"✅ 无问题文件: {files_ok}", flush=True)
    print(f"⚠️  有问题文件: {files_with_issues}", flush=True)
    print(f"总问题数: {total_issues}", flush=True)
    print(f"🔧 总修复数: {total_fixed}", flush=True)
    
    # 保存报告
    report_path = "check_and_fix_v2_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 报告已保存: {report_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

