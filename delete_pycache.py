import argparse
import fnmatch
import os
import shutil
import sys
from typing import Iterable, List, Optional


DEFAULT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".hypothesis",
    "htmlcov",
    ".tox",
    ".ipynb_checkpoints",
    ".cache",
    "build",
    "dist",
    ".coverage",  # Coverage directory
    "site-packages",  # Development site-packages
}

DEFAULT_DIR_PATTERNS = ["*.egg-info", "pip-wheel-metadata"]

DEFAULT_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".coverage",
    "coverage.xml",
    ".coverage.*",
    "*.log",
    ".DS_Store",
    "Thumbs.db",
]

DEFAULT_EXTRA_DIR_NAMES = {
    ".nox",
    ".ruff_cache",
    ".benchmarks",
}

DEFAULT_SKIP_TRAVERSE_DIRS = {".git", ".svn", ".hg"}

DEFAULT_EXTRA_FILE_PATTERNS = [
    "*.tmp",
    "*.temp",
    "*~",
    "*.swp",
    "*.swo",
    "*.bak",
    "*.orig",
    "#*#",  # Emacs backup files
    ".#*",  # Emacs lock files
]


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def delete_temporary_items(
    root_dir: str,
    dry_run: bool = False,
    verbose: bool = True,
    include_dir_patterns: Iterable[str] = (),
    include_file_patterns: Iterable[str] = (),
    exclude_dir_patterns: Iterable[str] = (),
    exclude_file_patterns: Iterable[str] = (),
    remove_node_modules: bool = False,
    remove_venvs: bool = False,
    venv_names: Iterable[str] = (".venv", "venv", ".env", "env"),
    follow_links: bool = False,
    max_depth: Optional[int] = None,
) -> dict:
    """Delete common temporary directories and files under root_dir.

    - Removes directories in DEFAULT_DIR_NAMES and those matching DEFAULT_DIR_PATTERNS.
    - Removes files matching DEFAULT_FILE_PATTERNS.
    - Removes directories matching patterns (like '*.egg-info').

    This function avoids descending into removed directories by modifying dirnames in-place.
    Returns statistics about what was removed.
    """
    stats = {"dirs_removed": 0, "files_removed": 0, "errors": 0}
    default_dir_names = set(DEFAULT_DIR_NAMES) | set(DEFAULT_EXTRA_DIR_NAMES)
    default_dir_patterns = list(DEFAULT_DIR_PATTERNS)
    default_file_patterns = list(DEFAULT_FILE_PATTERNS) + list(DEFAULT_EXTRA_FILE_PATTERNS)

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True, followlinks=follow_links):
        if max_depth is not None:
            rel = os.path.relpath(dirpath, root_dir)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirnames[:] = []
                continue

        # copy list because we may modify dirnames
        for d in list(dirnames):
            full_d = os.path.join(dirpath, d)
            
            # Skip version control and other protected directories
            if d in DEFAULT_SKIP_TRAVERSE_DIRS or matches_any(d, exclude_dir_patterns):
                dirnames.remove(d)
                continue

            should_remove_dir = (
                d in default_dir_names
                or matches_any(d, default_dir_patterns)
                or matches_any(d, include_dir_patterns)
                or (remove_node_modules and d == "node_modules")
                or (remove_venvs and d in set(venv_names))
            )

            if should_remove_dir:
                if dry_run:
                    if verbose:
                        print(f"Would remove directory: {full_d}")
                else:
                    try:
                        shutil.rmtree(full_d, ignore_errors=False)
                        stats["dirs_removed"] += 1
                        if verbose:
                            print(f"Removed directory: {full_d}")
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"Error removing directory {full_d}: {e}")
                # prevent os.walk from descending into it
                if d in dirnames:
                    dirnames.remove(d)

        # files
        for f in list(filenames):
            full_f = os.path.join(dirpath, f)
            if matches_any(f, exclude_file_patterns):
                continue

            if (
                matches_any(f, default_file_patterns)
                or matches_any(f, include_file_patterns)
            ):
                if dry_run:
                    if verbose:
                        print(f"Would remove file: {full_f}")
                else:
                    try:
                        os.remove(full_f)
                        stats["files_removed"] += 1
                        if verbose:
                            print(f"Removed file: {full_f}")
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"Error removing file {full_f}: {e}")
    
    return stats


def prune_empty_dirs(root_dir: str, dry_run: bool = False, verbose: bool = True):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # don't prune the root itself
        if dirpath == root_dir:
            continue
        try:
            if not os.listdir(dirpath):
                if dry_run:
                    if verbose:
                        print(f"Would remove empty directory: {dirpath}")
                else:
                    os.rmdir(dirpath)
                    if verbose:
                        print(f"Removed empty directory: {dirpath}")
        except Exception as e:
            print(f"Error pruning directory {dirpath}: {e}")


def clear_log_files(root_dir: str, delete_logs_dir: bool = False, dry_run: bool = False, verbose: bool = True):
    """Either truncate files under a `logs` dir, or delete the logs directory entirely.

    - If delete_logs_dir is True, the whole logs directory is removed.
    - If False, each file is truncated to 0 bytes.
    """
    log_dir = os.path.join(root_dir, "logs")

    if not os.path.exists(log_dir):
        if verbose:
            print(f"Log directory not found at: {log_dir}")
        return

    if delete_logs_dir:
        if dry_run:
            if verbose:
                print(f"Would remove logs directory: {log_dir}")
        else:
            try:
                shutil.rmtree(log_dir, ignore_errors=False)
                if verbose:
                    print(f"Removed logs directory: {log_dir}")
            except Exception as e:
                print(f"Error removing logs directory {log_dir}: {e}")
        return

    # otherwise truncate files inside logs
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if dry_run:
                    if verbose:
                        print(f"Would clear file: {file_path}")
                else:
                    with open(file_path, "w"):
                        pass
                    if verbose:
                        print(f"Cleared: {file_path}")
            except Exception as e:
                print(f"Error clearing {file_path}: {str(e)}")


def parse_args(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Remove temporary files and folders from a project tree")
    p.add_argument("root", nargs="?", default=None, help="Root directory to clean (defaults to script dir)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    p.add_argument("--yes", action="store_true", help="Don't prompt for confirmation")
    p.add_argument("--delete-logs", action="store_true", help="Remove the entire logs directory instead of truncating files")
    p.add_argument("--quiet", action="store_true", help="Reduce output")

    # Advanced controls
    p.add_argument("--include-dir", action="append", default=[], help="Additional directory patterns to remove (glob). Can be passed multiple times.")
    p.add_argument("--include-file", action="append", default=[], help="Additional file patterns to remove (glob). Can be passed multiple times.")
    p.add_argument("--exclude-dir", action="append", default=[], help="Directory patterns to exclude from deletion (glob). Can be passed multiple times.")
    p.add_argument("--exclude-file", action="append", default=[], help="File patterns to exclude from deletion (glob). Can be passed multiple times.")
    p.add_argument("--remove-node-modules", action="store_true", help="Also remove node_modules directories")
    p.add_argument("--remove-venvs", action="store_true", help="Also remove common virtualenv directories (.venv, venv, .env, env)")
    p.add_argument("--venv-names", nargs="*", default=None, help="Override names considered virtualenvs (space-separated)")
    p.add_argument("--follow-links", action="store_true", help="Follow symbolic links during traversal (use with caution)")
    p.add_argument("--max-depth", type=int, default=None, help="Limit traversal depth relative to root (1 = only root level)")
    p.add_argument("--prune-empty-dirs", action="store_true", help="Remove now-empty directories after cleanup")
    return p.parse_args(argv)


def main(argv: List[str] = None):
    args = parse_args(argv)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.abspath(args.root) if args.root else current_dir
    
    # Validate root directory exists
    if not os.path.exists(root_directory):
        print(f"Error: Root directory does not exist: {root_directory}")
        return 1
    
    if not os.path.isdir(root_directory):
        print(f"Error: Root path is not a directory: {root_directory}")
        return 1

    dry_run = args.dry_run
    verbose = not args.quiet

    if not args.yes and not dry_run:
        print(f"About to clean temporary files under: {root_directory}")
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted by user.")
            return 1

    if verbose:
        print("Starting cleanup process...")

    venv_names = args.venv_names if args.venv_names is not None else (".venv", "venv", ".env", "env")
    stats = delete_temporary_items(
        root_directory,
        dry_run=dry_run,
        verbose=verbose,
        include_dir_patterns=args.include_dir,
        include_file_patterns=args.include_file,
        exclude_dir_patterns=args.exclude_dir,
        exclude_file_patterns=args.exclude_file,
        remove_node_modules=args.remove_node_modules,
        remove_venvs=args.remove_venvs,
        venv_names=venv_names,
        follow_links=args.follow_links,
        max_depth=args.max_depth,
    )
    clear_log_files(root_directory, delete_logs_dir=args.delete_logs, dry_run=dry_run, verbose=verbose)

    if args.prune_empty_dirs:
        prune_empty_dirs(root_directory, dry_run=dry_run, verbose=verbose)

    if verbose:
        print(f"\nCleanup completed")
        print(f"Directories removed: {stats['dirs_removed']}")
        print(f"Files removed: {stats['files_removed']}")
        if stats['errors'] > 0:
            print(f"Errors encountered: {stats['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
