import pandas as pd
import re
import subprocess
from pathlib import Path
from tqdm import tqdm

print("=" * 100)
print("🔍 终极武器：直接解析原 PDF (PDFtoText)")
print("=" * 100)

df = pd.read_csv('FINAL_COMPLETE_Validated_V2.csv')

# 筛选目标：Ref < 20, 非书评, 非 Editorial
exclude_keywords = ['Book Review', 'Book review', 'Preface', 'Editorial', 'Introduction to', 'Commentary', 'Index', 'Contents', 'Foreword', 'Letter to Editor']
target_mask = (df['reference'] < 20) & (~df['file_name'].str.contains('|'.join(exclude_keywords), case=False)) & (df['subject'].isin(['Sociology', 'Psychology', 'Biology', 'Physics', 'Medicine', 'Computer Science', 'Business', 'Education', 'Environmental Science', 'Engineering']))

# 针对所有学科
targets = df[target_mask]
print(f"待解析论文数: {len(targets)}")

base_dir = Path("Dataset%20final/Dataset final")
updates = []

for idx, row in tqdm(targets.iterrows(), total=len(targets), desc="PDF解析中"):
    file_name = row['file_name']
    
    # 找 PDF (_origin.pdf)
    name_frag = file_name[:40]
    # 先尝试全名
    pdfs = list(base_dir.rglob(f"*{name_frag}*_origin.pdf"))
    target_pdf = pdfs[0] if pdfs else None
    
    if not target_pdf:
        # 尝试只要以 _origin.pdf 结尾
        pdfs = list(base_dir.rglob(f"*{name_frag}*.pdf"))
        # 过滤掉 layout, span, auto生成的
        valid_pdfs = [p for p in pdfs if 'layout' not in p.name and 'span' not in p.name and 'auto' in str(p.parent)]
        # 如果 auto 下有，优先 auto 下的任何 pdf
        if valid_pdfs:
            target_pdf = valid_pdfs[0]
            
    if target_pdf:
        try:
            # 使用 pdftotext
            # -layout 保持布局，更利于正则
            # 必须捕获输出
            result = subprocess.run(['pdftotext', '-layout', str(target_pdf), '-'], capture_output=True, text=True, timeout=60, errors='ignore')
            text = result.stdout
            
            if not text:
                continue
                
            # print(f"Processing {file_name}: {len(text)} chars")
            
            # 1. 寻找 References 区域
            # 同样使用宽泛标题 + 文末
            match = re.search(r'(References?|Bibliography|Literature Cited|Works Cited|Citations|Sources|Notes and References)', text, re.IGNORECASE)
            
            section = ""
            if match:
                # 尝试找到最后一个匹配 (通常 References 在最后)
                matches = list(re.finditer(r'(References?|Bibliography|Literature Cited|Works Cited)', text, re.IGNORECASE))
                if matches:
                    last_match = matches[-1]
                    # 只有当它出现在后半部分才算靠谱
                    if last_match.start() > len(text) * 0.4:
                        section = text[last_match.end():]
            
            if not section:
                section = text[-30000:] # 后3万字符
            
            # 2. 统计最大编号 (最可靠)
            max_vals = []
            
            # IEEE [n]
            # 有时 [1], [2] 会带空格 [ 1 ]
            ieee = [int(n) for n in re.findall(r'\[\s*(\d{1,4})\s*\]', section)]
            valid_ieee = [n for n in ieee if n < 2000]
            if valid_ieee: max_vals.append(('IEEE', max(valid_ieee)))
            
            # Paren (n)
            paren = [int(n) for n in re.findall(r'[\(（]\s*(\d{1,3})\s*[\)）]', section)]
            valid_paren = [n for n in paren if n < 1000] # 排除年份
            # 必须有一定数量
            if valid_paren and len(set(valid_paren)) > 5: max_vals.append(('Paren', max(valid_paren)))

            # Numbered Dot n. 
            # 必须行首
            dots = [int(n) for n in re.findall(r'^\s*(\d{1,4})\.\s+[A-Z]', section, re.MULTILINE)]
            if dots: max_vals.append(('Dot', max(dots)))
             
            # Numbered Space n Author
            space_nums = [int(n) for n in re.findall(r'^\s*(\d{1,4})\s+[A-Z]', section, re.MULTILINE)]
            if space_nums: max_vals.append(('Space', max(space_nums)))
            
            # APA (如果无编号)
            # 只能统计行数 (Author (Year))
            # 或者是行首是悬挂缩进? 很难识别。
            # 统计行首主要模式： A... (Year)
            apa_matches = re.findall(r'^[A-Z][a-zA-Z\s\.,&]+[\(（](19|20)\d{2}[a-z]?[\)）]', section, re.MULTILINE)
            if len(apa_matches) > 10: 
                max_vals.append(('APA', len(apa_matches)))
            
            if max_vals:
                best_type, best_max = max(max_vals, key=lambda x: x[1])
                
                # 更新条件：比原来大显著，且绝对值不可为个位数 (既然是 Review)
                threshold = max(row['reference'] * 1.5, 5)
                
                if best_max > threshold:
                    updates.append({
                        'index': idx,
                        'file_name': file_name,
                        'old_ref': row['reference'],
                        'new_ref': best_max,
                        'method': f"PDF_{best_type}"
                    })

        except Exception as e:
            # print(f"Error reading {file_name}: {e}")
            pass

print(f"\n✅ PDF 解析挽救了 {len(updates)} 篇论文！")

if updates:
    print("\n修复详情 (前20篇):")
    for up in updates[:20]:
        print(f"📄 {up['file_name'][:40]}... : {up['old_ref']} -> {up['new_ref']} ({up['method']})")
        df.at[up['index'], 'reference'] = up['new_ref']
        
    df.to_csv('FINAL_COMPLETE_Validated_PDF.csv', index=False)
    print("\n💾 已保存终极版: FINAL_COMPLETE_Validated_PDF.csv")
    
    # 学科汇总
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
    
    summary = df.groupby('subject')['reference'].mean().sort_values(ascending=False).round(1)
    print("\n📊 最终学科汇总:")
    print(summary)
    
    summary.to_csv('FINAL_SUBJECT_SUMMARY_Validated_PDF.csv')

else:
    print("\n没有新的发现。剩下的 Ref < 20 论文应该确实是短文。")
