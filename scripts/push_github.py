#!/usr/bin/env python3
"""把本地工作树推到 GitHub（Git Data API 路径，绕开 github.com:443 不可达）。

为什么不用 git push：
  本网络下 github.com:443 / ssh.github.com:443 都超时，只有 api.github.com 通
  （实测 0.8s）。API 的 blobs/trees/commits/refs 足以完成一次等价的 push。

行为：
  · 远端已有且 blob sha 相同的文件 **不重传**（git blob sha 直接可比）
  · 远端有、本地没有的路径 → 从树里删除（sha 置 null）
  · 新提交的 parent = 远端当前 HEAD；用 PATCH refs 快进
  · 推完逐文件复核：本地 `git hash-object` == 远端 tree 里该 path 的 sha

用法: .venv/bin/python scripts/push_github.py [--repo owner/name] [--branch main] [--dry]
"""
import argparse
import base64
import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gh(args, stdin_json=None):
    cmd = ['gh'] + args
    if stdin_json is not None:
        # 🚨 gh api 不读 stdin，必须显式 --input -，否则 body 为空 → HTTP 422 Validation Failed
        cmd += ['--input', '-']
    p = subprocess.run(cmd, cwd=PROJ, input=(json.dumps(stdin_json) if stdin_json else None),
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError('gh %s 失败: %s' % (' '.join(args[:4]), p.stderr[:400]))
    out = p.stdout.strip()
    return json.loads(out) if out.startswith(('{', '[')) else out


def local_files():
    # 🚨 必须 -z：默认 git ls-files 会把非 ASCII 路径转义成 "docs/\317\200..."，
    #    拿去 open() 直接 FileNotFoundError
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=PROJ,
                         capture_output=True).stdout.decode('utf-8')
    return [x for x in out.split('\x00') if x.strip()]


def blob_sha(path):
    return subprocess.run(['git', 'hash-object', path], cwd=PROJ,
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='kid14123000/AlphaZero_Gomoku_15x15_v24')
    ap.add_argument('--branch', default='main')
    ap.add_argument('--message', default=None)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    # 1. 远端现状
    ref = gh(['api', 'repos/%s/git/ref/heads/%s' % (a.repo, a.branch)])
    parent = ref['object']['sha']
    base_tree = gh(['api', 'repos/%s/git/commits/%s' % (a.repo, parent)])['tree']['sha']
    tree = gh(['api', 'repos/%s/git/trees/%s?recursive=1' % (a.repo, base_tree)])
    remote = {e['path']: e['sha'] for e in tree['tree'] if e['type'] == 'blob'}
    print('远端 HEAD %s · tree %s · %d 个文件' % (parent[:10], base_tree[:10], len(remote)))

    # 2. 本地文件与差异
    files = local_files()
    upload, same = [], 0
    for f in files:
        sha = blob_sha(f)
        if remote.get(f) == sha:
            same += 1
        else:
            upload.append((f, sha))
    deleted = [p for p in remote if p not in set(files)]
    print('本地 %d 个文件 · 未变 %d · 需上传 %d · 需删除 %d'
          % (len(files), same, len(upload), len(deleted)))

    if a.dry:
        for f, _ in upload:
            print('  +', f)
        for f in deleted:
            print('  -', f)
        return

    # 3. 上传 blob
    entries = []
    for i, (f, sha) in enumerate(upload, 1):
        raw = open(os.path.join(PROJ, f), 'rb').read()
        r = gh(['api', '--method', 'POST', 'repos/%s/git/blobs' % a.repo],
               stdin_json={'content': base64.b64encode(raw).decode(),
                           'encoding': 'base64'})
        if r['sha'] != sha:
            raise RuntimeError('blob sha 不一致: %s 本地 %s 远端 %s' % (f, sha, r['sha']))
        entries.append({'path': f, 'mode': '100644', 'type': 'blob', 'sha': r['sha']})
        print('  [%d/%d] %s (%.1f KB)' % (i, len(upload), f, len(raw) / 1024))
    for f in deleted:
        entries.append({'path': f, 'mode': '100644', 'type': 'blob', 'sha': None})

    if not entries:
        print('没有差异，无需提交')
        return

    # 4. tree → commit → ref
    new_tree = gh(['api', '--method', 'POST', 'repos/%s/git/trees' % a.repo],
                  stdin_json={'base_tree': base_tree, 'tree': entries})['sha']
    msg = a.message or 'sync: 本地仓库快照（Git Data API 推送）'
    commit = gh(['api', '--method', 'POST', 'repos/%s/git/commits' % a.repo],
                stdin_json={'message': msg, 'tree': new_tree, 'parents': [parent]})['sha']
    gh(['api', '--method', 'PATCH', 'repos/%s/git/refs/heads/%s' % (a.repo, a.branch)],
       stdin_json={'sha': commit, 'force': True})
    print('✅ 新提交 %s (tree %s)' % (commit, new_tree))

    # 5. 逐文件复核 sha
    tree2 = gh(['api', 'repos/%s/git/trees/%s?recursive=1' % (a.repo, commit)])
    got = {e['path']: e['sha'] for e in tree2['tree'] if e['type'] == 'blob'}
    bad = [f for f in files if got.get(f) != blob_sha(f)]
    extra = [p for p in got if p not in set(files)]
    print('复核: 远端 %d 个文件 · 内容不一致 %d · 多余 %d' % (len(got), len(bad), len(extra)))
    if bad:
        print('  ✗ 不一致:', bad[:8])
        sys.exit(1)
    print('✅ 逐文件 blob sha 全部一致')


if __name__ == '__main__':
    main()
