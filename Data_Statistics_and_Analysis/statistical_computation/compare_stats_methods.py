#!/usr/bin/env python3
"""
对比旧方法和新方法的统计结果
"""

import re
from pathlib import Path
from improved_statistics import (
    count_citations_improved,
    count_references_improved,
    count_images_improved,
    count_equations_improved,
    count_outline_items_improved
)

# 测试文件
test_file = Path("Dataset%20final/Dataset final/Sociology/Sociology_84_Organizing_School-to-Work_Transition_Research_from_a_Sustainable_Career_Perspective__A_Review_and_Research_Agenda/Sociology_84_Organizing_School-to-Work_Transition_Research_from_a_Sustainable_Career_Perspective__A_Review_and_Research_Agenda/auto/Sociology_84_Organizing_School-to-Work_Transition_Research_from_a_Sustainable_Career_Perspective__A_Review_and_Research_Agenda.md")

print("=" * 80)
print("📊 统计方法对比测试")
print("=" * 80)

# 读取文件
with open(test_file, 'r', encoding='utf-8') as f:
    full_text = f.read()

print(f"\n测试文件: {test_file.name[:60]}...")
print(f"文件大小: {len(full_text):,} 字符")

# ========== Citations统计对比 ==========
print("\n" + "=" * 80)
print("1️⃣  CITATIONS 统计对比")
print("=" * 80)

# 旧方法
old_citations = 0
old_patterns = [
    r'\[\d+\]',
    r'\[\d+,\s*\d+\]',
    r'\(\w+\s+\d{4}\)',
    r'\b[A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\)'
]
for pattern in old_patterns:
    matches = re.findall(pattern, full_text)
    old_citations += len(matches)
    print(f"  旧方法 - 模式 '{pattern[:30]}...' : {len(matches)} 个")

print(f"\n  🔴 旧方法总计: {old_citations} 个citation")

# 新方法
new_citations = count_citations_improved(full_text)
print(f"  🟢 新方法总计: {new_citations} 个citation (去重后)")
print(f"  📈 差异: {old_citations - new_citations} 个重复项被消除")

# ========== References统计对比 ==========
print("\n" + "=" * 80)
print("2️⃣  REFERENCES 统计对比")
print("=" * 80)

# 旧方法（不准确）
old_refs_pattern = r'^\[\d+\]|\d+\.\s+[A-Z]'
old_refs = len(re.findall(old_refs_pattern, full_text, re.MULTILINE))
print(f"  🔴 旧方法: {old_refs} 个reference (误判章节标题等)")

# 新方法
new_refs = count_references_improved(full_text)
print(f"  🟢 新方法: {new_refs} 个reference (基于章节识别)")
print(f"  📈 准确性提升: {abs(old_refs - new_refs)} 个误判被修正")

# ========== 手动验证参考文献章节 ==========
ref_section_match = re.search(r'#\s*References?\s*\n', full_text, re.IGNORECASE)
if ref_section_match:
    ref_section = full_text[ref_section_match.end():ref_section_match.end()+500]
    print(f"\n  📝 参考文献章节预览:")
    print(f"  {ref_section[:200].strip()}...")

# ========== Equations统计对比 ==========
print("\n" + "=" * 80)
print("3️⃣  EQUATIONS 统计对比")
print("=" * 80)

# 旧方法（可能重复）
old_equations = 0
equation_patterns = [
    r'\$\$.*?\$\$',
    r'\$.*?\$',
    r'\\begin\{equation\}',
    r'\\begin\{align\}',
]
for pattern in equation_patterns[:2]:  # 只测试前两个
    matches = re.findall(pattern, full_text, re.DOTALL)
    old_equations += len(matches)
    print(f"  旧方法 - 模式 '{pattern}': {len(matches)} 个")

print(f"\n  🔴 旧方法总计: {old_equations} 个equation (可能重复)")

# 新方法
new_equations = count_equations_improved(full_text)
print(f"  🟢 新方法总计: {new_equations} 个equation (去重后)")
print(f"  📈 差异: {old_equations - new_equations} 个重复项")

# ========== Outline统计对比 ==========
print("\n" + "=" * 80)
print("4️⃣  OUTLINE 统计对比")
print("=" * 80)

# 旧方法（包括列表项）
old_outline = 0
lines = full_text.split('\n')
for line in lines:
    if re.match(r'^\d+\.|\*\s+|\-\s+|•\s+', line.strip()):
        old_outline += 1
print(f"  🔴 旧方法: {old_outline} 个outline (包括所有列表项)")

# 新方法（只包括标题）
new_outline = count_outline_items_improved(full_text)
print(f"  🟢 新方法: {new_outline} 个outline (只包括章节标题)")
print(f"  📈 差异: {old_outline - new_outline} 个列表项被排除")

# ========== 总结 ==========
print("\n" + "=" * 80)
print("📋 改进总结")
print("=" * 80)

improvements = [
    ("Citations", old_citations, new_citations, "去重，避免重复计数"),
    ("References", old_refs, new_refs, "章节识别，避免误判"),
    ("Equations", old_equations, new_equations, "优先级处理，避免重叠"),
    ("Outline", old_outline, new_outline, "只统计标题，排除列表"),
]

for metric, old_val, new_val, reason in improvements:
    change_pct = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
    symbol = "📉" if new_val < old_val else "📈"
    print(f"\n{metric:15} {old_val:5} → {new_val:5}  {symbol} {change_pct:+6.1f}%")
    print(f"                原因: {reason}")

print("\n" + "=" * 80)
print("✅ 统计方法对比完成!")
print("=" * 80)
