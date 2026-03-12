import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

def advanced_ref_extract(text):
    # 1. 尝试更宽泛的标题
    # 包括：Reference(s), Bibliography, Literature Cited, Sources, Citations
    # 忽略大小写
    title_match = re.search(r'#+\s*(References?|Bibliography|Literature Cited|Works Cited|Citations|Sources|Notes and References)', text, re.IGNORECASE)
    
    section = ""
    if title_match:
        section = text[title_match.end():]
    else:
        # 如果没标题，截取最后 20% 的内容作为潜在区域
        # 或者是最后 10000 字符
        section = text[-15000:]
        
    # 2. 统计各种格式
    
    # IEEE [1]
    ieee = len(set(re.findall(r'\[\d+\]', section)))
    
    # Numbered Dot 1. (行首)
    num_dot = len(set(re.findall(r'^\d+\.', section, re.MULTILINE)))
    
    # Numbered Space 1 Author (行首)
    num_space = len(set(re.findall(r'^\d+\s+[A-Z]', section, re.MULTILINE)))
    
    # Paren (1)
    # 排除年份，只取 < 300
    parens = [int(p) for p in re.findall(r'[\(（](\d{1,3})[\)）]', section) if int(p) < 300]
    num_paren = len(set(parens)) if parens else 0
    
    # Numbered Dot Inline (1. Author... 2. Author...)
    # 这种最难，因为容易匹配到正文数字。需要验证序列性。
    # 查找所有 "数字. 大写"
    dot_inline = [int(n) for n in re.findall(r'(\d+)\.\s*[A-Z]', section)]
    num_dot_inline = 0
    if dot_inline:
        # 必须有一定的序列性
        # 比如找到 1, 2, 3, ... 或者至少有一些连续的
        # 简单判定：去重后数量 > 10 且 max < count * 3
        uniq = sorted(list(set(dot_inline)))
        if len(uniq) > 5 and uniq[-1] < len(uniq) * 3 + 20: 
            num_dot_inline = len(uniq)
            
    # 取最大值
    counts = {
        'IEEE': ieee,
        'Dot': num_dot,
        'Space': num_space,
        'Paren': num_paren,
        'DotInline': num_dot_inline
    }
    
    return max(counts.values()), max(counts, key=counts.get)

def main():
    print("=" * 80)
    print("🔧 修复剩余 Ref=0 论文 (宽泛标题 + 文末扫描)")
    print("=" * 80)
    
    # 读取当前数据
    df = pd.read_csv('FINAL_COMPLETE_RECALCULATED.csv')
    
    # 只处理 Ref=0 或极少的 (<5)
    target_indices = df[df['reference'] < 5].index
    print(f"待检查论文数: {len(target_indices)}")
    
    base_dir = Path("Dataset%20final/Dataset final")
    
    updates = []
    
    for idx in tqdm(target_indices, desc="扫描中"):
        row = df.loc[idx]
        file_name = row['file_name']
        
        # 找 MD 文件
        name_frag = file_name[:40]
        md_files = list(base_dir.rglob(f"*{name_frag}*.md"))
        target_md = next((f for f in md_files if f.parent.name == 'auto'), md_files[0] if md_files else None)
        
        if target_md:
            with open(target_md, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            count, method = advanced_ref_extract(text)
            
            # 只有当找到显著数量 (>5) 时才更新
            if count > 5:
                updates.append({
                    'index': idx,
                    'file_name': file_name,
                    'old_ref': row['reference'],
                    'new_ref': count,
                    'method': method
                })
    
    print(f"\n成功修复: {len(updates)} 篇")
    
    if updates:
        print("\n修复详情:")
        for up in updates:
            print(f"- {up['file_name'][:40]}... : {up['old_ref']} -> {up['new_ref']} ({up['method']})")
            
            # 应用更新
            df.at[up['index'], 'reference'] = up['new_ref']
            
        # 保存
        df.to_csv('FINAL_COMPLETE_RECALCULATED_V2.csv', index=False)
        print("\n💾 已保存 V2 数据: FINAL_COMPLETE_RECALCULATED_V2.csv")
        
        # 生成汇总
        import re
        def extract_subject(discipline):
            if pd.isna(discipline): return "Unknown"
            s = str(discipline).split('_')[0]
            match = re.match(r'([A-Za-z\s]+)', s)
            if match: s = match.group(0).strip()
            return s

        df['subject'] = df['discipline'].apply(extract_subject)
        df['subject'] = df['subject'].replace({
            'Phychology': 'Psychology',
            'Physchology': 'Psychology',
            'Computer': 'Computer Science',
            'Computer science': 'Computer Science'
        })
        
        col_map = {'img': 'Img', 'tab': 'Tab', 'eq': 'Eq', 'para': 'Para', 'words': 'Words', 'sent': 'Sent', 'citation': 'Citation', 'reference': 'Reference'}
        agg_dict = {k: 'mean' for k in col_map.keys() if k in df.columns}
        summary = df.groupby('subject').agg(agg_dict).round(1)
        summary = summary.rename(columns=col_map)
        summary = summary.sort_values('Reference', ascending=False)
        
        print("\n📊 最新学科汇总:")
        print(summary['Reference'])
        summary.to_csv('FINAL_SUBJECT_SUMMARY_RECALCULATED_V2.csv')

if __name__ == "__main__":
    main()
