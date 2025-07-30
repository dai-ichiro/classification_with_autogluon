from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import shutil
import fnmatch

class FileWatcher:
    def __init__(self, watch_dir, dest_dir, file_patterns=None, recursive=False):
        self.watch_dir = Path(watch_dir)
        self.dest_dir = Path(dest_dir)
        self.file_patterns = file_patterns or ['*']
        self.recursive = recursive
        self.observer = None
        
        # 必要なディレクトリを作成
        self.dest_dir.mkdir(parents=True, exist_ok=True)
    
    def _matches_pattern(self, filename):
        """ファイル名がパターンにマッチするかチェック"""
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.file_patterns)
    
    def start(self):
        """ファイル監視を開始"""
        if self.observer is not None:
            print("File watcher is already running")
            return
        
        handler = FileWatcherHandler(
            dest_dir=self.dest_dir,
            file_patterns=self.file_patterns,
            callback=self.on_file_created
        )
        
        self.observer = Observer()
        self.observer.start()
        self.observer.schedule(handler, str(self.watch_dir), recursive=self.recursive)
        
        print(f"📂 File monitoring started: {self.watch_dir} -> {self.dest_dir}")
        print(f"📋 Patterns: {self.file_patterns}")
    
    def stop(self):
        """ファイル監視を停止"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("🛑 File monitoring stopped")
    
    def on_file_created(self, src_path, dest_path):
        """ファイル作成時のコールバック（オーバーライド可能）"""
        print(f"🔍 File detected: {src_path.name}")
        print(f"📁 Backup created: {dest_path}")

class FileWatcherHandler(FileSystemEventHandler):
    def __init__(self, dest_dir, file_patterns, callback=None):
        self.dest_dir = dest_dir
        self.file_patterns = file_patterns
        self.callback = callback
    
    def _matches_pattern(self, filename):
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.file_patterns)
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        src_path = Path(event.src_path)
        
        if not self._matches_pattern(src_path.name):
            return
        
        dest_path = self.dest_dir / src_path.name
        
        try:
            shutil.copy2(src_path, dest_path)
            if self.callback:
                self.callback(src_path, dest_path)
        except Exception as e:
            print(f"❌ Error copying {src_path.name}: {e}")