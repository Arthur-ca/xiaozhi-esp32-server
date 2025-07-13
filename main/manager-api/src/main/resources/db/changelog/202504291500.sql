-- 创建用户对话记录表
CREATE TABLE IF NOT EXISTS ai_chat_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
    device_id VARCHAR(64) NOT NULL COMMENT '设备ID',
    client_id VARCHAR(64) NOT NULL COMMENT '客户端ID',
    role ENUM('user', 'assistant') NOT NULL COMMENT '发言角色(用户/助手)',
    content TEXT NOT NULL COMMENT '消息内容',
    question_type VARCHAR(32) DEFAULT NULL COMMENT '问题类型/分类',
    suggestion TEXT DEFAULT NULL COMMENT '修改建议',
    is_modified BOOLEAN DEFAULT FALSE COMMENT '是否已修改',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_session (session_id),
    INDEX idx_device (device_id),
    INDEX idx_client (client_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户对话历史记录表';