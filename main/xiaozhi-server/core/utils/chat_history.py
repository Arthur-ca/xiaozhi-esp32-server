import uuid
from datetime import datetime
import pymysql
import yaml
import os
import asyncio
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

class ChatHistoryManager:
    def __init__(self, config):
        self.config = config
        self.connection = None
        logger.bind(tag=TAG).info("初始化 ChatHistoryManager")
        self.db_config = self._load_db_config()
        logger.bind(tag=TAG).info(f"数据库配置加载成功: {self.db_config}")
        self.save_queue = asyncio.Queue()
        self.save_task = None

    def _load_db_config(self):
        """从application-dev.yml加载数据库配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                     'manager-api', 'src', 'main', 'resources', 'application-dev.yml')
            logger.bind(tag=TAG).info(f"尝试加载配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                db_config = config['spring']['datasource']['druid']
                logger.bind(tag=TAG).info("成功加载数据库配置")
                return db_config
        except Exception as e:
            logger.bind(tag=TAG).error(f"加载数据库配置失败: {str(e)}")
            raise

    def _get_connection(self):
        if self.connection is None or not self.connection.open:
            try:
                logger.bind(tag=TAG).info("尝试建立数据库连接")
                # 修复数据库URL解析
                url = self.db_config['url']
                # 移除jdbc:mysql://前缀
                url = url.replace('jdbc:mysql://', '')
                # 分割主机和数据库部分
                host_port, db_part = url.split('/', 1)
                host = host_port.split(':')[0]
                # 获取数据库名（移除查询参数）
                database = db_part.split('?')[0]
                logger.bind(tag=TAG).info(f"连接数据库: {host}/{database}")
                
                # 确保密码是字符串类型
                password = str(self.db_config['password'])
                
                self.connection = pymysql.connect(
                    host=host,
                    user=self.db_config['username'],
                    password=password,
                    database=database,
                    charset='utf8mb4'
                )
                logger.bind(tag=TAG).info("数据库连接成功")
            except Exception as e:
                logger.bind(tag=TAG).error(f"数据库连接失败: {str(e)}")
                raise
        return self.connection

    async def save_chat_history(self, user_id, agent_id, device_id, messages):
        """
        异步保存对话历史到数据库
        :param user_id: 用户ID
        :param agent_id: 智能体ID
        :param device_id: 设备ID
        :param messages: 消息列表
        """
        logger.bind(tag=TAG).info(f"准备保存对话历史 - 用户ID: {user_id}, 消息数量: {len(messages)}")
        # 将保存任务放入队列
        await self.save_queue.put((user_id, agent_id, device_id, messages))
        logger.bind(tag=TAG).info("保存任务已加入队列")
        
        # 如果保存任务还没有启动，则启动它
        if self.save_task is None or self.save_task.done():
            self.save_task = asyncio.create_task(self._process_save_queue())
            logger.bind(tag=TAG).info("保存任务已启动")

    async def _process_save_queue(self):
        """处理保存队列中的任务"""
        logger.bind(tag=TAG).info("开始处理保存队列")
        while True:
            try:
                # 从队列中获取保存任务
                user_id, agent_id, device_id, messages = await self.save_queue.get()
                logger.bind(tag=TAG).info(f"从队列获取到保存任务 - 用户ID: {user_id}")
                
                # 在线程池中执行数据库操作
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._save_to_database,
                    user_id,
                    agent_id,
                    device_id,
                    messages
                )
                
                # 标记任务完成
                self.save_queue.task_done()
                logger.bind(tag=TAG).info(f"保存任务处理完成 - 用户ID: {user_id}")
                
            except Exception as e:
                logger.bind(tag=TAG).error(f"处理保存队列失败: {str(e)}")
                # 如果队列为空，退出循环
                if self.save_queue.empty():
                    logger.bind(tag=TAG).info("保存队列为空，退出处理")
                    break

    def _save_to_database(self, user_id, agent_id, device_id, messages):
        """实际的数据库保存操作"""
        conn = None
        try:
            logger.bind(tag=TAG).info(f"开始保存到数据库 - 用户ID: {user_id}")
            logger.bind(tag=TAG).info(f"原始user_id长度: {len(str(user_id))}")
            conn = self._get_connection()
            with conn.cursor() as cursor:
                # 创建新的对话历史记录，使用更短的ID格式
                chat_id = str(uuid.uuid4()).replace('-', '')[:32]
                # 使用完整的user_id
                now = datetime.now()
                
                # 插入对话历史记录
                logger.bind(tag=TAG).info(f"插入对话历史记录 - chat_id: {chat_id}")
                cursor.execute("""
                    INSERT INTO ai_chat_history 
                    (id, user_id, agent_id, device_id, message_count, creator, create_date, updater, update_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    chat_id,
                    user_id,
                    agent_id,
                    device_id,
                    len(messages),
                    user_id,
                    now,
                    user_id,
                    now
                ))

                # 插入对话消息
                logger.bind(tag=TAG).info(f"开始插入对话消息 - 消息数量: {len(messages)}")
                for msg in messages:
                    if msg.role in ['user', 'assistant']:
                        msg_id = str(uuid.uuid4()).replace('-', '')[:32]
                        logger.bind(tag=TAG).debug(f"插入消息 - ID: {msg_id}, 角色: {msg.role}")
                        cursor.execute("""
                            INSERT INTO ai_chat_message 
                            (id, user_id, chat_id, role, content, creator, create_date, updater, update_date)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            msg_id,
                            user_id,
                            chat_id,
                            msg.role,
                            msg.content,
                            user_id,
                            now,
                            user_id,
                            now
                        ))

            conn.commit()
            logger.bind(tag=TAG).info(f"成功保存对话历史 - 用户ID: {user_id}")
            return True
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存对话历史失败: {str(e)}")
            if conn:
                conn.rollback()
                logger.bind(tag=TAG).info("已回滚事务")
            return False
        finally:
            if conn:
                conn.close()
                self.connection = None
                logger.bind(tag=TAG).info("数据库连接已关闭") 