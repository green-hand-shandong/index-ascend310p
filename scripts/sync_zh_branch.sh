#!/usr/bin/env bash
# 从 main 分支生成/更新 main-zh 分支（GitCode 默认分支：中文主页）
# main:    README.md = EN（链接 README_zh.md）, README_zh.md = ZH
# main-zh: README.md = ZH（链接 README_en.md）, README_en.md = EN，其余文件与 main 一致
# 用法：先在 main 提交所有变更，再运行本脚本，然后推送两个分支。
set -e
cd "$(dirname "$0")/.."
git checkout main
git branch -D main-zh 2>/dev/null || true
git checkout -b main-zh
cp README.md README_en.md          # 先保存英文版（此时 README.md 还是 main 的英文内容）
cp README_zh.md README.md          # 中文版成为主页
# 修正中文版内的英文链接指向
sed -i 's|\[English\](README\.md)|[English](README_en.md)|g; s|\[README\.md\](README\.md)|[README_en.md](README_en.md)|g' README.md
# README_en.md 内部的中文链接改为指向 README.md
sed -i 's|\[简体中文\](README_zh\.md)|[简体中文](README.md)|g; s|\[README_zh\.md\](README_zh\.md)|[README.md](README.md)|g' README_en.md
git add -A
git diff --cached --quiet && echo "main-zh 无变化" || git commit -q -m "GitCode 中文主页（与 main 同步）"
git checkout main
echo "main-zh 就绪"
