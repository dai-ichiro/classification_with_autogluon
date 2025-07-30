from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import time
import shutil
from functools import wraps
import fnmatch

class AdvancedFileWatcherHandler(FileSystemEventHandler):
    def __init__(self, dest_dir, callback=None, file_patterns=None, copy_files=True):
        self.dest_dir = Path(dest_dir)
        self.callback = callback
        self.file_patterns = file_patterns or ['*']
        self.copy_files = copy_files
    
    def _matches_pattern(self, filename):
        """ファイル名がパターンにマッチするかチェック"""
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.file_patterns)
    
    def on_created(self, event):
        if not event.is_directory:
            src_path = Path(event.src_path)
            
            # パターンマッチングをチェック
            if not self._matches_pattern(src_path.name):
                return
            
            dest_path = self.dest_dir / src_path.name
            
            try:
                # ファイルコピーが有効な場合のみコピー
                if self.copy_files:
                    # 宛先ディレクトリが存在しない場合は作成
                    self.dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {src_path.name} to {self.dest_dir}")
                
                # コールバック関数があれば実行
                if self.callback:
                    self.callback(src_path, dest_path if self.copy_files else None)
                    
            except Exception as e:
                print(f"Error processing {src_path.name}: {e}")

class FileWatcherConfig:
    def __init__(
        self,
        watch_dir: str,
        dest_dir: str = None,
        recursive: bool = False,
        file_patterns: list[str] = None,
        copy_files: bool = True,
        daemon: bool = False
    ):
        self.watch_dir = watch_dir
        self.dest_dir = dest_dir
        self.recursive = recursive
        self.file_patterns = file_patterns or ['*']
        self.copy_files = copy_files
        self.daemon = daemon

def watch_files(config: FileWatcherConfig):
    """
    設定オブジェクトを使用するファイル監視デコレーター
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if config.copy_files and not config.dest_dir:
                raise ValueError("copy_files=True requires dest_dir to be specified")
            
            handler = AdvancedFileWatcherHandler(
                dest_dir=config.dest_dir or "/tmp",
                callback=func,
                file_patterns=config.file_patterns,
                copy_files=config.copy_files
            )
            
            observer = Observer()
            observer.schedule(handler, str(config.watch_dir), recursive=config.recursive)
            
            if config.daemon:
                # デーモンモードで監視を開始
                observer.start()
                print(f"Daemon file watcher started: monitoring {config.watch_dir}")
                return func(*args, **kwargs)
            else:
                # 同期モードで監視
                observer.start()
                print(f"File watcher started: monitoring {config.watch_dir}")
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    observer.stop()
                    observer.join()
                    print("File watcher stopped")
        
        def start_monitoring():
            """継続的な監視を開始"""
            if config.copy_files and not config.dest_dir:
                raise ValueError("copy_files=True requires dest_dir to be specified")
                
            handler = AdvancedFileWatcherHandler(
                dest_dir=config.dest_dir or "/tmp",
                callback=func,
                file_patterns=config.file_patterns,
                copy_files=config.copy_files
            )
            
            observer = Observer()
            observer.schedule(handler, str(config.watch_dir), recursive=config.recursive)
            observer.start()
            
            print(f"Monitoring {config.watch_dir}...")
            print(f"File patterns: {config.file_patterns}")
            print("Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
                print("\nStopping file watcher...")
            
            observer.join()
        
        wrapper.start_monitoring = start_monitoring
        return wrapper
    
    return decorator

# 便利な設定済みデコレーター
def simple_file_watcher(watch_dir, dest_dir, recursive=False):
    """シンプルなファイル監視デコレーター（元のコードに近い動作）"""
    config = FileWatcherConfig(
        watch_dir=watch_dir,
        dest_dir=dest_dir,
        recursive=recursive,
        copy_files=True
    )
    return watch_files(config)

def file_monitor_only(watch_dir, file_patterns=None, recursive=False):
    """ファイル監視のみ（コピーしない）"""
    config = FileWatcherConfig(
        watch_dir=watch_dir,
        dest_dir=None,
        recursive=recursive,
        file_patterns=file_patterns,
        copy_files=False
    )
    return watch_files(config)

# 使用例
if __name__ == "__main__":
    
    # 例1: シンプルな使用（元のコードと同じ動作）
    @simple_file_watcher("/path/to/source", "/path/to/destination")
    def handle_new_file(src_path, dest_path):
        print(f"New file processed: {src_path.name}")
    
    # 例2: 特定のファイルタイプのみ監視
    @file_monitor_only("/path/to/source", file_patterns=["*.txt", "*.log"])
    def handle_text_files(src_path, dest_path):
        print(f"Text file detected: {src_path.name}")
        # ファイルの内容を読んで処理
        with open(src_path, 'r') as f:
            content = f.read()
            print(f"File content length: {len(content)} characters")
    
    # 例3: 高度な設定
    config = FileWatcherConfig(
        watch_dir="/path/to/source",
        dest_dir="/path/to/destination",
        recursive=True,
        file_patterns=["*.pdf", "*.doc*"],
        copy_files=True,
        daemon=False
    )
    
    @watch_files(config)
    def handle_documents(src_path, dest_path):
        print(f"Document file processed: {src_path.name}")
        # ドキュメント固有の処理をここに追加
    
    # 監視を開始
    handle_new_file.start_monitoring()