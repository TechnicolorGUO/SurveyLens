#!/usr/bin/env python3
"""
诊断脚本 - 找出导致卡住的论文
"""

import re
import time
from pathlib import Path
from improved_statistics import count_citations_improved

def test_single_paper(paper_path):
    """测试单篇论文的处理速度"""
    try:
        start_time = time.time()
        
        with open(paper_path, 'r', encoding='utf-8', errors='replace') as f:
            full_text = f.read()
        
        read_time = time.time() - start_time
        file_size = len(full_text)
        
        # 测试citations统计（最可能卡住的地方）
        cit_start = time.time()
        citations = count_citations_improved(full_text)
        cit_time = time.time() - cit_start
        
        return {
            'path': str(paper_path),
            'name': paper_path.stem[:50],
            'size': file_size,
            'read_time': read_time,
            'citation_time': cit_time,
            'total_time': read_time + cit_time,
            'citations': citations,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'path': str(paper_path),
            'name': paper_path.stem[:50],
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    print("=" * 100)
    print("🔍 诊断卡住的论文")
    print("=" * 100)
    
    # 查找所有论文
    base_dir = Path("Dataset%20final/Dataset final")
    all_papers = list(base_dir.rglob("*.md"))
    all_papers = [p for p in all_papers if not p.name.startswith('.') and p.parent.name == 'auto']
    
    print(f"\n总论文数: {len(all_papers)}")
    print(f"已知卡在: 984/1000")
    print(f"需要检查的: 最后{len(all_papers) - 984}篇\n")
    
    # 只测试980-1000之间的论文
    test_range = all_papers[980:]
    
    print(f"测试范围: 论文 #{981} 到 #{len(all_papers)}")
    print("=" * 100)
    
    results = []
    for i, paper in enumerate(test_range, 981):
        print(f"\n[{i}/{len(all_papers)}] 测试: {paper.stem[:60]}...")
        
        result = test_single_paper(paper)
        results.append(result)
        
        if result['status'] == 'OK':
            print(f"  ✅ 成功")
            print(f"     文件大小: {result['size']:,} 字符")
            print(f"     读取时间: {result['read_time']:.2f}秒")
            print(f"     Citations: {result['citations']} 个 (耗时 {result['citation_time']:.2f}秒)")
            print(f"     总耗时: {result['total_time']:.2f}秒")
            
            # 标记慢速文件
            if result['total_time'] > 5:
                print(f"  ⚠️  这篇论文处理很慢！({result['total_time']:.1f}秒)")
            if result['size'] > 500000:
                print(f"  ⚠️  这是一个超大文件！({result['size']:,} 字符)")
        else:
            print(f"  ❌ 错误: {result['error']}")
    
    # 汇总
    print("\n" + "=" * 100)
    print("📊 诊断汇总")
    print("=" * 100)
    
    slow_papers = [r for r in results if r['status'] == 'OK' and r['total_time'] > 5]
    large_papers = [r for r in results if r['status'] == 'OK' and r['size'] > 500000]
    error_papers = [r for r in results if r['status'] == 'ERROR']
    
    if slow_papers:
        print(f"\n⚠️  慢速论文 (>5秒): {len(slow_papers)} 篇")
        for r in sorted(slow_papers, key=lambda x: x['total_time'], reverse=True):
            print(f"   - {r['name'][:50]}")
            print(f"     总耗时: {r['total_time']:.1f}秒 | 大小: {r['size']:,} | Citations: {r['citations']}")
    
    if large_papers:
        print(f"\n⚠️  超大文件 (>500KB): {len(large_papers)} 篇")
        for r in sorted(large_papers, key=lambda x: x['size'], reverse=True):
            print(f"   - {r['name'][:50]}")
            print(f"     大小: {r['size']:,} 字符 | 耗时: {r['total_time']:.1f}秒")
    
    if error_papers:
        print(f"\n❌ 错误论文: {len(error_papers)} 篇")
        for r in error_papers:
            print(f"   - {r['name'][:50]}: {r['error']}")
    
    # 找出最可能卡住的文件
    if results:
        slowest = max([r for r in results if r['status'] == 'OK'], 
                     key=lambda x: x.get('total_time', 0))
        print(f"\n🐌 处理最慢的论文:")
        print(f"   文件: {slowest['name']}")
        print(f"   路径: {slowest['path']}")
        print(f"   大小: {slowest['size']:,} 字符")
        print(f"   耗时: {slowest['total_time']:.1f}秒")
        print(f"   Citations统计: {slowest['citation_time']:.1f}秒")
    
    print("\n" + "=" * 100)
    print("✅ 诊断完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
