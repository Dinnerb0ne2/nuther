# Nuther API 使用指南

## 目录

1. [快速开始](#快速开始)
2. [核心 API](#核心-api)
3. [模块 API](#模块-api)
4. [使用示例](#使用示例)
5. [高级用法](#高级用法)

---

## 快速开始

### 基础使用

```python
from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel
from src.chat import ChatBot

# 创建词典
vocab = Vocabulary(vocab_size=1000)
vocab.build_vocab(["hello", "world", "how", "are", "you"])

# 创建模型
model = NutherModel(
    vocab=vocab,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    use_memory=True
)

# 存储知识
model.store_knowledge("This is some knowledge about the world.")

# 创建聊天机器人
chat_bot = ChatBot(model)

# 开始对话
chat_bot.interactive_chat()
```

### 单次生成

```python
response = model.generate("Hello", max_length=20, temperature=0.8)
print(f"Response: {response}")
```

---

## 核心 API

### NutherFramework

框架管理器，协调所有模块。

#### 初始化

```python
from src.main import NutherFramework

# 创建新模型
framework = NutherFramework()

# 加载已有模型
framework = NutherFramework(
    model_path="data/model",
    vocab_path="data/vocab.txt"
)
```

#### 方法

##### build_vocabulary

构建词典。

```python
texts = [
    "Hello world",
    "How are you?",
    "I am fine"
]
framework.build_vocabulary(texts)
```

##### load_knowledge_base

加载知识到记忆库。

```python
framework.load_knowledge_base("knowledge.txt")
```

##### crawl_and_store

爬取网站并存储知识。

```python
urls = ["https://example.com"]
framework.crawl_and_store(urls, max_pages=10)
```

##### chat

启动交互式对话。

```python
framework.chat(
    max_length=100,
    temperature=0.8
)
```

##### generate

生成单次响应。

```python
response = framework.generate(
    input_text="Hello",
    max_length=50,
    temperature=0.8
)
```

##### save

保存模型。

```python
framework.save("data/saved_model")
```

##### get_statistics

获取统计信息。

```python
stats = framework.get_statistics()
print(stats)
```

---

### NutherModel

核心模型类。

#### 初始化

```python
from src.model import NutherModel

model = NutherModel(
    vocab=vocab,
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    cell_dim=None,
    encoder_bidirectional=False,
    decoder_use_moe=True,
    num_experts=config.NUM_EXPERTS,
    top_k=config.TOP_K_EXPERTS,
    use_memory=True
)
```

#### 方法

##### forward

前向传播。

```python
import numpy as np

# 编码输入
input_indices = vocab.text_to_indices("Hello", add_start=False, add_end=False)
input_indices = np.array([input_indices], dtype=np.int32)

# 教师强制模式
target_indices = vocab.text_to_indices("Hi there", add_start=True, add_end=True)
target_indices = np.array([target_indices], dtype=np.int32)

result = model.forward(input_indices, target_indices=target_indices)
output_logits = result['output_logits']

# 自回归生成模式
result = model.forward(input_indices, max_output_length=20)
output_indices = result['output_indices']
```

##### generate

生成响应。

```python
response = model.generate(
    input_text="Hello",
    max_length=100,
    temperature=0.8
)
```

##### generate_with_memory

带记忆的生成。

```python
response, memory_context = model.generate_with_memory(
    input_text="Tell me about neural networks",
    max_length=100,
    temperature=0.8
)
```

##### chat

对话交互。

```python
result = model.chat(
    user_input="How do you work?",
    max_length=100,
    temperature=0.8
)

print(f"User: {result['user_input']}")
print(f"Response: {result['response']}")
print(f"Memory: {result.get('memory_context', '')}")
```

##### store_knowledge

存储知识。

```python
model.store_knowledge(
    text="Neural networks are computational models inspired by biological brains.",
    metadata={'source': 'textbook', 'topic': 'neural_networks'}
)
```

##### store_knowledge_from_file

从文件加载知识。

```python
model.store_knowledge_from_file(
    "data/knowledge.txt",
    metadata={'source': 'file'}
)
```

##### save / load

保存和加载模型。

```python
# 保存
model.save("data/my_model")

# 加载
model.load("data/my_model")
```

##### get_parameter_count

获取参数统计。

```python
params = model.get_parameter_count()
print(f"Encoder: {params['encoder']:,}")
print(f"Decoder: {params['decoder']:,}")
print(f"Total: {params['total']:,}")
```

---

### Vocabulary

词典类。

#### 初始化

```python
from src.vocab import Vocabulary

vocab = Vocabulary(vocab_size=10000)
```

#### 方法

##### build_vocab

构建词典。

```python
texts = [
    "Hello world",
    "This is a test",
    "Machine learning is awesome"
]
vocab.build_vocab(texts, min_freq=2)
```

##### text_to_indices

文本转索引。

```python
indices = vocab.text_to_indices(
    text="Hello world",
    add_start=True,
    add_end=True,
    max_length=50
)
```

##### indices_to_text

索引转文本。

```python
text = vocab.indices_to_text(
    indices,
    skip_special=True
)
```

##### tokenize

分词。

```python
tokens = vocab.tokenize("Hello, world!")
# ['hello', ',', 'world', '!']
```

##### get_vocab_size

获取词典大小。

```python
size = vocab.get_vocab_size()
```

##### save / load

保存和加载词典。

```python
vocab.save("data/vocab.txt")
vocab.load("data/vocab.txt")
```

---

### MemoryBank

记忆库类。

#### 初始化

```python
from src.memory import MemoryBank

memory_bank = MemoryBank(
    memory_id="my_memory",
    max_chunks=1000,
    top_k=5,
    threshold=0.5,
    keyword_weight=0.3
)
```

#### 方法

##### store

存储文本。

```python
chunk_ids = memory_bank.store(
    text="This is some text to store in memory.",
    metadata={'source': 'user'}
)
```

##### store_dialogue_turn

存储对话轮次。

```python
turn_id = memory_bank.store_dialogue_turn(
    user_input="Hello",
    model_output="Hi there!",
    turn_id=0
)
```

##### retrieve

检索记忆。

```python
results = memory_bank.retrieve(
    query="neural networks",
    top_k=5
)

for chunk, similarity in results:
    print(f"[{similarity:.3f}] {chunk.get_text()}")
```

##### retrieve_with_keywords

基于关键词检索。

```python
results = memory_bank.retrieve_by_keywords(
    query="machine learning",
    min_keyword_matches=2,
    top_k=3
)
```

##### get_context

获取上下文。

```python
context = memory_bank.get_context(
    query="neural networks",
    max_context_length=500,
    top_k=3
)
```

##### get_augmented_input

获取增强输入。

```python
augmented = memory_bank.get_augmented_input(
    user_input="Tell me about AI",
    max_context_length=500,
    include_conversation=True
)
```

##### get_statistics

获取统计信息。

```python
stats = memory_bank.get_statistics()
print(stats)
```

##### save / load

保存和加载。

```python
memory_bank.save("data/memory_bank.json")
memory_bank.load("data/memory_bank.json")
```

---

### ChatBot

聊天机器人类。

#### 初始化

```python
from src.chat import ChatBot

chat_bot = ChatBot(
    model=model,
    max_history=50
)
```

#### 方法

##### create_session

创建会话。

```python
session_id = chat_bot.create_session("session_1")
```

##### chat

对话。

```python
result = chat_bot.chat(
    user_input="Hello",
    session_id="session_1",
    max_length=100,
    temperature=0.8,
    use_memory=True
)

print(f"Response: {result['response']}")
```

##### batch_chat

批量对话。

```python
inputs = ["Hello", "How are you?", "Tell me about AI"]
responses = chat_bot.batch_chat(inputs, session_id="session_1")

for response in responses:
    print(response['response'])
```

##### interactive_chat

交互式对话。

```python
chat_bot.interactive_chat(
    session_id="session_1",
    max_length=100,
    temperature=0.8
)
```

##### switch_session

切换会话。

```python
chat_bot.switch_session("session_2")
```

##### get_session

获取会话。

```python
session = chat_bot.get_session("session_1")
print(session.get_statistics())
```

---

## 模块 API

### LSTM 模块

#### LSTMCell

```python
from src.lstm import LSTMCell

cell = LSTMCell(input_dim=128, hidden_dim=256)

# 前向传播
x = np.random.randn(32, 128).astype(np.float32)
h_prev = np.zeros((32, 256), dtype=np.float32)
c_prev = np.zeros((32, 256), dtype=np.float32)

h_next, c_next, cache = cell.forward(x, h_prev, c_prev)
```

#### EmbeddingLSTM

```python
from src.lstm import EmbeddingLSTM

embed_lstm = EmbeddingLSTM(
    vocab_size=1000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2
)

indices = np.random.randint(0, 1000, size=(32, 10))
output, h_final, c_final = embed_lstm.forward_inference(indices)
```

### MoE 模块

#### MoE

```python
from src.moe import MoE

moe = MoE(
    input_dim=256,
    output_dim=512,
    num_experts=8,
    gating_type='top_k',
    expert_type='feed_forward'
)

x = np.random.randn(32, 256).astype(np.float32)
output, gating_weights, expert_outputs = moe.forward(x)
```

#### SparseMoE

```python
from src.moe import SparseMoE

sparse_moe = SparseMoE(
    input_dim=256,
    output_dim=512,
    num_experts=8,
    top_k=2,
    load_balance_weight=0.01
)

output, gating_weights, load_balance_loss = sparse_moe.forward(x)
```

### 爬虫模块

#### CrawlerPipeline

```python
from src.crawler import CrawlerPipeline

crawler = CrawlerPipeline()

# 爬取并存储
stored_count = crawler.crawl_and_store(
    urls=["https://example.com"],
    max_pages=10
)

# 获取知识库
kb = crawler.get_knowledge_bank()
corpus = kb.get_text_corpus()
```

#### 创建示例知识库

```python
from src.crawler import create_sample_knowledge_base

kb = create_sample_knowledge_base()
```

---

## 使用示例

### 示例 1: 基础对话

```python
from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel

# 创建词典
vocab = Vocabulary(vocab_size=1000)
vocab.build_vocab(["hello", "world", "how", "are", "you"])

# 创建模型
model = NutherModel(
    vocab=vocab,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2
)

# 对话
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    response = model.generate(user_input, max_length=20)
    print(f"Bot: {response}")
```

### 示例 2: 知识问答

```python
from src.config import config
from src.vocab import Vocabulary
from src.model import NutherModel

vocab = Vocabulary(vocab_size=1000)
vocab.build_vocab(["neural", "network", "learning", "ai"])

model = NutherModel(vocab=vocab, use_memory=True)

# 存储知识
model.store_knowledge(
    "Neural networks are computational models inspired by biological brains."
)
model.store_knowledge(
    "Deep learning uses neural networks with multiple layers."
)

# 问答
questions = [
    "What are neural networks?",
    "Tell me about deep learning."
]

for question in questions:
    response, memory_context = model.generate_with_memory(question)
    print(f"Q: {question}")
    print(f"A: {response}")
    print(f"Memory: {memory_context[:50]}...")
    print()
```

### 示例 3: 多会话管理

```python
from src.chat import ChatBot

chat_bot = ChatBot(model)

# 创建多个会话
session1 = chat_bot.create_session("user1")
session2 = chat_bot.create_session("user2")

# 会话 1
chat_bot.chat("Hello", session_id=session1)
chat_bot.chat("How are you?", session_id=session1)

# 会话 2
chat_bot.chat("Hi", session_id=session2)
chat_bot.chat("What's your name?", session_id=session2)

# 查看会话统计
for session_id in [session1, session2]:
    session = chat_bot.get_session(session_id)
    stats = session.get_statistics()
    print(f"{session_id}: {stats}")
```

### 示例 4: 批量生成

```python
inputs = [
    "Hello",
    "How are you?",
    "Tell me about AI",
    "What can you do?"
]

responses = chat_bot.batch_chat(
    inputs,
    max_length=20,
    temperature=0.8
)

for i, response in enumerate(responses):
    print(f"Input: {inputs[i]}")
    print(f"Output: {response['response']}")
    print()
```

---

## 高级用法

### 自定义专家

```python
from src.moe import Expert, MoE

class CustomExpert(Expert):
    def __init__(self, input_dim, output_dim, expert_id):
        super().__init__(input_dim, output_dim, expert_id)
        # 自定义初始化
    
    def forward(self, x):
        # 自定义前向传播
        return output
    
    def get_parameters(self):
        return params
    
    def set_parameters(self, params):
        pass
    
    def reset_parameters(self):
        pass

# 使用自定义专家
custom_experts = [
    CustomExpert(256, 512, i) for i in range(8)
]
```

### 自定义相似度计算

```python
from src.memory import SimilarityCalculator

class CustomSimilarityCalculator(SimilarityCalculator):
    def compute_similarity(self, query, chunk):
        # 自定义相似度计算
        return similarity

# 使用自定义计算器
from src.memory import MemoryBank
sim_calc = CustomSimilarityCalculator(keyword_weight=0.5)
memory_bank = MemoryBank()
memory_bank.retriever.similarity_calculator = sim_calc
```

### 模型微调

```python
# 获取参数
params = model.get_parameters()

# 修改参数
params['encoder']['embedding_matrix'] *= 0.1

# 设置参数
model.set_parameters(params)
```

### 记忆管理

```python
# 清空记忆
model.get_memory_bank().clear_memory()

# 清空对话历史
model.get_memory_bank().clear_conversation_history()

# 清空所有
model.get_memory_bank().clear_all()
```

### 配置调优

```python
from src.config import config

# 修改配置
config.EMBEDDING_DIM = 512
config.HIDDEN_DIM = 1024
config.NUM_EXPERTS = 16
config.TOP_K_EXPERTS = 4

# 创建新模型
model = NutherModel(vocab=vocab)
```

---

## 最佳实践

1. **词典构建**
   - 使用足够大的文本语料
   - 设置合理的 min_freq
   - 保存词典以供重用

2. **知识存储**
   - 分块存储长文本
   - 添加有用的元数据
   - 定期清理记忆库

3. **对话管理**
   - 为不同用户创建独立会话
   - 限制会话历史长度
   - 定期保存对话记录

4. **性能优化**
   - 使用批量处理
   - 限制记忆检索范围
   - 调整 Top-K 参数

5. **错误处理**
   - 捕获并处理异常
   - 提供友好的错误信息
   - 记录日志

---

## 常见问题

### Q: 如何提高生成质量？
A: 
- 增加训练数据
- 调整温度参数
- 优化知识库内容
- 使用更大的模型

### Q: 如何减少内存使用？
A:
- 减少 MEMORY_SIZE
- 定期清理记忆
- 使用更小的嵌入维度

### Q: 如何加快推理速度？
A:
- 减少 TOP_K_EXPERTS
- 使用更小的模型
- 限制记忆检索范围

### Q: 如何处理未知词汇？
A:
- 使用 UNK 令牌
- 增大词典大小
- 添加更多训练数据

---

## 总结

Nuther 提供了丰富的 API 用于构建对话式神经网络应用。通过合理使用这些 API，可以快速实现各种对话场景。建议从简单示例开始，逐步探索更高级的功能。