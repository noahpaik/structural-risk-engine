# -*- coding: utf-8 -*-
"""
Emoji 제거 스크립트
"""
import re

# back.py 읽기
with open('back.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 이모지 매핑
emoji_map = {
    '📊': '[INFO]',
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARN]',
    '🔧': '[DATA]',
    '🚀': '[START]',
    '🤖': '[AI]',
    '🎯': '[TARGET]',
    '🔍': '[SEARCH]',
    '💼': '[POSITION]',
    '⚖️': '[BALANCE]',
    '📈': '[METRICS]',
    '📅': '[DATE]',
    '🟢': '[NORMAL]',
    '🟡': '[ELEVATED]',
    '🔴': '[HIGH]'
}

# 교체
for emoji, replacement in emoji_map.items():
    content = content.replace(emoji, replacement)

# 저장
with open('back.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Emoji removal completed!")
