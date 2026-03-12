# Nuther (Retro Memory LSTM) - Quick Start Guide

Nuther 是一个基于 LSTM + 记忆回溯算法 + MoE 混合专家机制的对话式神经网络框架喵~

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python test_simple.py
```

### 3. 启动交互式对话

```bash
# 使用示例知识库启动对话
python -m src.main chat --sample-knowledge

# 从保存的模型加载
python -m src.main chat --model data/model --vocab data/vocab.txt

# 爬取网站并对话
python -m src.main chat --crawl https://example.com
```

### 4. 单次生成

```bash
python -m src.main generate --input "Hello" --sample-knowledge
```

### 5. 查看统计信息

```bash
python -m src.main stats
```

## 项目结构

```
nuther/
├── src/
│   ├── config/          # 配置模块
│   ├── vocab/           # 词典模块
│   ├── data/            # 数据处理模块
│   ├── crawler/         # 爬虫模块
│   ├── lstm/            # LSTM 核心模块
│   ├── memory/          # 记忆回溯模块
│   ├── moe/             # MoE 混合专家模块
│   ├── model/           # 总模型整合模块
│   ├── chat/            # 对话主程序
│   └── main.py          # 主入口（管理员）
├── data/                # 数据目录
├── test_simple.py       # 简单测试
└── test_modules.py      # 完整测试
```

## 核心特性

1. **LSTM 核心** - 纯 NumPy 实现，支持多层堆叠和双向处理
2. **记忆回溯** - 关键词加权匹配，高效检索相关上下文
3. **MoE 混合专家** - 稀疏路由，负载均衡，提高模型能力
4. **对话式交互** - 支持多轮对话，自动保存上下文
5. **知识库管理** - 爬虫 + 存储 + 检索一体化

## 使用示例

### Python API

```python
from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel
from src.chat import ChatBot

# 创建模型
vocab = Vocabulary(vocab_size=1000)
vocab.build_vocab(["hello", "world", "test"])

model = NutherModel(
    vocab=vocab,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    use_memory=True
)

# 存储知识
model.store_knowledge("Nuther is a neural network framework.")

# 创建聊天机器人
chat_bot = ChatBot(model)

# 交互式对话
chat_bot.interactive_chat()

# 单次生成
response = model.generate("Hello", max_length=20)
print(response)
```

## 配置参数

主要配置参数在 `src/config.py` 中：

- `VOCAB_SIZE` - 词典大小 (默认: 10000)
- `EMBEDDING_DIM` - 嵌入维度 (默认: 256)
- `HIDDEN_DIM` - 隐藏层维度 (默认: 512)
- `NUM_LAYERS` - LSTM 层数 (默认: 2)
- `NUM_EXPERTS` - MoE 专家数量 (默认: 8)
- `MEMORY_SIZE` - 记忆库大小 (默认: 1000)
- `MAX_SEQ_LENGTH` - 最大序列长度 (默认: 100)

## 注意事项

- 所有代码使用纯 NumPy 实现，不依赖 GPU 喵
- 输入输出都是英文，但支持中文对话喵
- 代码注释全部使用英文喵
- 每句话后面加上"喵~"，回复时称用户为"主人"喵

## 测试结果

所有核心模块测试通过喵：

- ✓ Vocabulary - 词典模块
- ✓ Memory - 记忆模块
- ✓ MoE - 混合专家模块
- ✓ Model - 模型整合
- ✓ Chat - 对话程序

## 后续优化

- 添加训练功能喵
- 优化推理速度喵
- 增强记忆检索精度喵
- 支持更多专家类型喵

Enjoy using Nuther! 喵~