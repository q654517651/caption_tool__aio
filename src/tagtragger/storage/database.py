"""
Database manager - SQLite数据库管理器
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from ..utils.logger import log_info, log_error
from ..utils.exceptions import StorageError
from ..config import get_config

class Database:
    """SQLite数据库管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.db_path = Path(self.config.storage.workspace_root) / "data.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            with self.get_connection() as conn:
                # 数据集表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS datasets (
                        dataset_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        dataset_type TEXT DEFAULT 'image',
                        description TEXT DEFAULT '',
                        created_time TEXT NOT NULL,
                        modified_time TEXT NOT NULL,
                        image_count INTEGER DEFAULT 0,
                        labeled_count INTEGER DEFAULT 0,
                        tags TEXT DEFAULT '[]'
                    )
                ''')
                
                # 训练任务表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS training_tasks (
                        task_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        training_type TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        state TEXT DEFAULT 'pending',
                        progress REAL DEFAULT 0.0,
                        current_step INTEGER DEFAULT 0,
                        total_steps INTEGER DEFAULT 0,
                        current_epoch INTEGER DEFAULT 0,
                        loss REAL DEFAULT 0.0,
                        learning_rate REAL DEFAULT 0.0,
                        eta_seconds INTEGER NULL,
                        speed REAL NULL,
                        created_time TEXT NOT NULL,
                        started_time TEXT NULL,
                        completed_time TEXT NULL,
                        error_message TEXT DEFAULT '',
                        output_dir TEXT DEFAULT '',
                        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                    )
                ''')
                
                # 设置表
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_time TEXT NOT NULL
                    )
                ''')
                
                # 创建索引
                conn.execute('CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_training_tasks_dataset_id ON training_tasks (dataset_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_training_tasks_state ON training_tasks (state)')
                
                conn.commit()
                
            log_info("数据库初始化完成")
            
        except Exception as e:
            log_error(f"数据库初始化失败: {str(e)}")
            raise StorageError(f"数据库初始化失败: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 启用字典式访问
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise StorageError(f"数据库连接错误: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    # === 数据集相关操作 ===
    
    def save_dataset(self, dataset_dict: Dict[str, Any]) -> bool:
        """保存数据集信息"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO datasets (
                        dataset_id, name, dataset_type, description, created_time, 
                        modified_time, image_count, labeled_count, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    dataset_dict['dataset_id'],
                    dataset_dict['name'],
                    dataset_dict.get('dataset_type', 'image'),
                    dataset_dict.get('description', ''),
                    dataset_dict['created_time'],
                    dataset_dict['modified_time'],
                    len(dataset_dict.get('images', {})),
                    len([l for l in dataset_dict.get('images', {}).values() if l.strip()]),
                    json.dumps(dataset_dict.get('tags', []))
                ))
                conn.commit()
                return True
                
        except Exception as e:
            log_error(f"保存数据集失败: {str(e)}")
            return False
    
    def load_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """加载数据集信息"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    'SELECT * FROM datasets WHERE dataset_id = ?', 
                    (dataset_id,)
                ).fetchone()
                
                if row:
                    return {
                        'dataset_id': row['dataset_id'],
                        'name': row['name'],
                        'dataset_type': row['dataset_type'],
                        'description': row['description'],
                        'created_time': row['created_time'],
                        'modified_time': row['modified_time'],
                        'tags': json.loads(row['tags'] or '[]'),
                        'images': {}  # 图片信息从文件系统加载
                    }
                return None
                
        except Exception as e:
            log_error(f"加载数据集失败: {str(e)}")
            return None
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """获取所有数据集列表"""
        try:
            with self.get_connection() as conn:
                rows = conn.execute('''
                    SELECT dataset_id, name, dataset_type, description, 
                           created_time, modified_time, image_count, labeled_count
                    FROM datasets 
                    ORDER BY modified_time DESC
                ''').fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            log_error(f"获取数据集列表失败: {str(e)}")
            return []
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        try:
            with self.get_connection() as conn:
                conn.execute('DELETE FROM datasets WHERE dataset_id = ?', (dataset_id,))
                conn.execute('DELETE FROM training_tasks WHERE dataset_id = ?', (dataset_id,))
                conn.commit()
                return True
                
        except Exception as e:
            log_error(f"删除数据集失败: {str(e)}")
            return False
    
    # === 训练任务相关操作 ===
    
    def save_training_task(self, task_dict: Dict[str, Any]) -> bool:
        """保存训练任务"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO training_tasks (
                        task_id, name, training_type, dataset_id, config_json,
                        state, progress, current_step, total_steps, current_epoch,
                        loss, learning_rate, eta_seconds, speed,
                        created_time, started_time, completed_time, error_message, output_dir
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_dict['task_id'],
                    task_dict['config']['name'],
                    task_dict['config']['training_type'],
                    task_dict['config']['dataset_id'],
                    json.dumps(task_dict['config']),
                    task_dict['state'],
                    task_dict['progress'],
                    task_dict['current_step'],
                    task_dict['total_steps'],
                    task_dict['current_epoch'],
                    task_dict['loss'],
                    task_dict['learning_rate'],
                    task_dict.get('eta_seconds'),
                    task_dict.get('speed'),
                    task_dict['created_time'],
                    task_dict.get('started_time'),
                    task_dict.get('completed_time'),
                    task_dict.get('error_message', ''),
                    task_dict.get('output_dir', '')
                ))
                conn.commit()
                return True
                
        except Exception as e:
            log_error(f"保存训练任务失败: {str(e)}")
            return False
    
    def load_training_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """加载训练任务"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    'SELECT * FROM training_tasks WHERE task_id = ?', 
                    (task_id,)
                ).fetchone()
                
                if row:
                    return {
                        'task_id': row['task_id'],
                        'config': json.loads(row['config_json']),
                        'state': row['state'],
                        'progress': row['progress'],
                        'current_step': row['current_step'],
                        'total_steps': row['total_steps'],
                        'current_epoch': row['current_epoch'],
                        'loss': row['loss'],
                        'learning_rate': row['learning_rate'],
                        'eta_seconds': row['eta_seconds'],
                        'speed': row['speed'],
                        'created_time': row['created_time'],
                        'started_time': row['started_time'],
                        'completed_time': row['completed_time'],
                        'error_message': row['error_message'],
                        'output_dir': row['output_dir'],
                        'checkpoint_files': [],
                        'sample_images': []
                    }
                return None
                
        except Exception as e:
            log_error(f"加载训练任务失败: {str(e)}")
            return None
    
    def list_training_tasks(self) -> List[Dict[str, Any]]:
        """获取所有训练任务列表"""
        try:
            with self.get_connection() as conn:
                rows = conn.execute('''
                    SELECT t.task_id, t.name, t.training_type, t.dataset_id, t.state,
                           t.progress, t.created_time, t.started_time, t.completed_time,
                           d.name as dataset_name
                    FROM training_tasks t
                    LEFT JOIN datasets d ON t.dataset_id = d.dataset_id
                    ORDER BY t.created_time DESC
                ''').fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            log_error(f"获取训练任务列表失败: {str(e)}")
            return []
    
    def delete_training_task(self, task_id: str) -> bool:
        """删除训练任务"""
        try:
            with self.get_connection() as conn:
                conn.execute('DELETE FROM training_tasks WHERE task_id = ?', (task_id,))
                conn.commit()
                return True
                
        except Exception as e:
            log_error(f"删除训练任务失败: {str(e)}")
            return False
    
    # === 设置相关操作 ===
    
    def save_setting(self, key: str, value: Any) -> bool:
        """保存设置"""
        try:
            from datetime import datetime
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_time)
                    VALUES (?, ?, ?)
                ''', (key, json.dumps(value), datetime.now().isoformat()))
                conn.commit()
                return True
                
        except Exception as e:
            log_error(f"保存设置失败 {key}: {str(e)}")
            return False
    
    def load_setting(self, key: str, default: Any = None) -> Any:
        """加载设置"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    'SELECT value FROM settings WHERE key = ?', 
                    (key,)
                ).fetchone()
                
                if row:
                    return json.loads(row['value'])
                return default
                
        except Exception as e:
            log_error(f"加载设置失败 {key}: {str(e)}")
            return default
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            with self.get_connection() as conn:
                dataset_count = conn.execute('SELECT COUNT(*) FROM datasets').fetchone()[0]
                task_count = conn.execute('SELECT COUNT(*) FROM training_tasks').fetchone()[0]
                
                # 获取数据库文件大小
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'database_path': str(self.db_path),
                    'database_size_mb': round(db_size / 1024 / 1024, 2),
                    'dataset_count': dataset_count,
                    'training_task_count': task_count
                }
                
        except Exception as e:
            log_error(f"获取数据库信息失败: {str(e)}")
            return {}
    
    def vacuum_database(self) -> bool:
        """优化数据库"""
        try:
            with self.get_connection() as conn:
                conn.execute('VACUUM')
                conn.commit()
                log_info("数据库优化完成")
                return True
                
        except Exception as e:
            log_error(f"数据库优化失败: {str(e)}")
            return False