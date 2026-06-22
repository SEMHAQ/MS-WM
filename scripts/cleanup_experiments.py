"""
清理实验目录 - 将旧实验文件移动到备份目录
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / 'experiments'
BACKUP_DIR = ROOT / 'experiments_backup'


def cleanup():
    """清理实验目录"""
    print("=" * 60)
    print("清理实验目录")
    print("=" * 60)

    # 创建备份目录
    BACKUP_DIR.mkdir(exist_ok=True)

    # 要保留的目录
    keep_dirs = {'paper'}

    # 要保留的文件模式
    keep_patterns = {'.gitkeep'}

    # 统计
    moved_count = 0
    kept_count = 0

    # 遍历实验目录
    for item in EXPERIMENTS_DIR.iterdir():
        # 跳过目录
        if item.is_dir():
            if item.name in keep_dirs:
                print(f"保留目录: {item.name}/")
                kept_count += 1
            else:
                # 移动到备份
                dest = BACKUP_DIR / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
                print(f"移动目录: {item.name}/ -> experiments_backup/{item.name}/")
                moved_count += 1
            continue

        # 跳过保留的文件
        if item.name in keep_patterns:
            kept_count += 1
            continue

        # 移动文件到备份
        dest = BACKUP_DIR / item.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))
        moved_count += 1

    print(f"\n完成!")
    print(f"保留: {kept_count} 项")
    print(f"移动: {moved_count} 项")
    print(f"备份位置: {BACKUP_DIR}")


if __name__ == '__main__':
    cleanup()
