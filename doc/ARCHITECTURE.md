# Nuther 架构设计文档

## 系统架构

### 整体架构

Nuther 采用分层架构设计，从底层到上层依次为：

```
┌─────────────────────────────────────────────────┐
│               应用层 (Application)               │
│         - 命令行接口 - 交互式对话 - API          │
├─────────────────────────────────────────────────┤
│               协调层 (Coordination)              │
│            - 框架管理器 - 会话管理               │
├─────────────────────────────────────────────────┤
│               模型层 (Model)                     │
│      - 编码器 - 解码器 - 记忆检索 - MoE          │
├─────────────────────────────────────────────────┤
│               组件层 (Components)                │
│   - LSTM - Memory - MoE - Vocabulary - Crawler  │
├─────────────────────────────────────────────────┤
│               数据层 (Data)                      │
│        - 数据处理 - 知识库 - 配置管理            │
└─────────────────────────────────────────────────┘
```

## 模块详细设计

### 1. 配置模块 (Config Module)

**职责**: 集中管理所有配置参数

**主要类**:
- `ModelConfig`: 模型配置类

**配置项**:
```python
# 模型架构
VOCAB_SIZE = 10000              # 词典大小
EMBEDDING_DIM = 256             # 嵌入维度
HIDDEN_DIM = 512                # 隐藏层维度
CELL_DIM = 512                  # 细胞状态维度
NUM_LAYERS = 2                  # LSTM 层数

# 记忆配置
MEMORY_SIZE = 1000              # 记忆容量
MEMORY_DIM = 256                # 记忆向量维度
RETRIEVAL_TOP_K = 5             # 检索 Top-K
SIMILARITY_THRESHOLD = 0.5      # 相似度阈值
KEYWORD_WEIGHT = 0.3            # 关键词权重

# MoE 配置
NUM_EXPERTS = 8                 # 专家数量
EXPERT_HIDDEN_DIM = 256         # 专家隐藏维度
TOP_K_EXPERTS = 2              # Top-K 专家
```

**设计特点**:
- 单例模式
- 类型安全
- 默认值合理
- 易于扩展

---

### 2. 词典模块 (Vocabulary Module)

**职责**: 文本和索引之间的双向转换

**主要类**:
- `Vocabulary`: 词典类

**核心功能**:
```python
# 词典构建
build_vocab(texts, min_freq=2)

# 文本到索引
text_to_indices(text, add_start, add_end, max_length)

# 索引到文本
indices_to_text(indices, skip_special)

# 分词
tokenize(text)
```

**数据结构**:
```python
word2idx: Dict[str, int]      # 单词到索引
idx2word: Dict[int, str]      # 索引到单词
word_counts: Counter          # 词频统计
```

**特殊令牌**:
- `<PAD>`: 填充令牌 (ID: 0)
- `<START>`: 开始令牌 (ID: 1)
- `<END>`: 结束令牌 (ID: 2)
- `<UNK>`: 未知令牌 (ID: 3)

**分词策略**:
- 支持英文和中文
- 保留基本标点
- 过滤特殊字符
- 去除停用词

---

### 3. LSTM 模块 (LSTM Module)

#### 3.1 LSTM Cell

**职责**: 单时间步的 LSTM 计算

**门控机制**:
```
遗忘门: f = σ(W_f · [h_{t-1}, x_t] + b_f)
输入门: i = σ(W_i · [h_{t-1}, x_t] + b_i)
输出门: o = σ(W_o · [h_{t-1}, x_t] + b_o)
候选状态: g = tanh(W_g · [h_{t-1}, x_t] + b_g)

细胞状态: C_t = f ⊙ C_{t-1} + i ⊙ g
隐藏状态: h_t = o ⊙ tanh(C_t)
```

**参数结构**:
```python
# 组合权重矩阵 (4 * cell_dim)
W: [input_dim + hidden_dim, 4 * cell_dim]
b: [4 * cell_dim]

# 分离权重
Wf, Wi, Wo, Wg: [input_dim + hidden_dim, cell_dim]
bf, bi, bo, bg: [cell_dim]
```

**初始化**:
- Xavier 初始化
- 避免梯度消失
- 稳定训练

#### 3.2 LSTM Layer

**职责**: 处理序列输入

**功能**:
- 序列前向传播
- 隐藏状态传递
- 可配置输出方式

**输出模式**:
- `return_sequences=True`: 返回所有时间步输出
- `return_sequences=False`: 只返回最后时间步输出

#### 3.3 LSTM Network

**职责**: 堆叠多层 LSTM

**特性**:
- 多层堆叠
- 双向支持
- 状态管理

#### 3.4 Embedding LSTM

**职责**: 文本序列处理

**组件**:
- 嵌入层: token → vector
- LSTM 层: 序列编码
- 投影层: 输出投影

**参数量计算**:
```
嵌入层: vocab_size × embedding_dim
LSTM 层: 4 × (input_dim + hidden_dim) × hidden_dim × num_layers
```

---

### 4. 记忆模块 (Memory Module)

#### 4.1 Memory Chunk

**职责**: 单个记忆块

**数据结构**:
```python
{
    'chunk_id': str,           # 唯一标识
    'content': str,            # 文本内容
    'keywords': Dict[str, float],  # 关键词权重
    'content_hash': str,       # 内容哈希
    'metadata': Dict           # 元数据
}
```

**关键词提取**:
- TF-IDF 计算
- 停用词过滤
- 权重归一化
- Top-K 选择

#### 4.2 Memory Storage

**职责**: 记忆存储管理

**功能**:
- 添加记忆块
- 去重检测
- LRU 淘汰
- 统计信息

**存储策略**:
- 哈希去重
- 访问计数
- 最近访问时间
- 容量限制

#### 4.3 Similarity Calculator

**职责**: 计算相似度

**相似度公式**:
```
similarity = keyword_weight × keyword_sim + semantic_weight × semantic_sim

# 关键词相似度
keyword_sim = Σ(w_query × w_chunk) / |matched_keywords|

# 语义相似度 (Jaccard)
semantic_sim = |tokens_query ∩ tokens_chunk| / |tokens_query ∪ tokens_chunk|
```

**关键词权重**:
- 基于词频
- 归一化处理
- 匹配加权

#### 4.4 Memory Retriever

**职责**: 记忆检索

**检索策略**:
- Top-K 选择
- 阈值过滤
- 关键词匹配
- 解释生成

**检索流程**:
```
1. 提取查询关键词
2. 计算所有记忆块相似度
3. 过滤低于阈值的块
4. 选择 Top-K
5. 返回排序结果
```

#### 4.5 Memory Bank

**职责**: 记忆库整合

**功能**:
- 存储管理
- 检索服务
- 对话历史
- 上下文增强

**上下文增强**:
```
augmented_input = memory_context + conversation_history + user_input
```

---

### 5. MoE 模块 (MoE Module)

#### 5.1 Expert

**职责**: 专家模型

**专家类型**:
- `FeedForwardExpert`: 前馈网络专家
- `LSTMExpert`: LSTM 专家

**FeedForwardExpert 结构**:
```
Input → Hidden1 → ReLU → Hidden2 → ReLU → Output
```

**LSTMExpert 结构**:
```
Input → LSTM Cell → Hidden State → Projection → Output
```

#### 5.2 Gating Network

**职责**: 门控路由

**门控类型**:
- `TopKGating`: Top-K 稀疏门控
- `SoftGating`: 软门控
- `GumbelSoftmaxGating`: Gumbel-Softmax 门控

**Top-K Gating 流程**:
```
1. 计算所有专家的 logits
2. 选择 Top-K 值
3. 置零其他值
4. Softmax 归一化
5. 返回稀疏权重
```

**门控网络结构**:
```
Input → Hidden → ReLU → Logits → Top-K → Softmax → Weights
```

#### 5.3 MoE

**职责**: MoE 整合

**前向传播**:
```
1. 计算门控权重: weights = gating(input)
2. 所有专家计算: outputs[i] = expert[i](input)
3. 加权求和: output = Σ(weights[i] × outputs[i])
```

**输出维度**:
- 输入: (batch_size, input_dim)
- 门控权重: (batch_size, num_experts)
- 专家输出: (batch_size, num_experts, output_dim)
- 最终输出: (batch_size, output_dim)

#### 5.4 Sparse MoE

**职责**: 稀疏 MoE 带负载均衡

**负载均衡损失**:
```
loss_balance = mean((expert_fraction - 1/num_experts)²)
loss_total = loss_task + λ × loss_balance
```

**专家利用率**:
```
utilization[i] = selection_count[i] / (total_samples × top_k)
```

---

### 6. 模型模块 (Model Module)

#### 6.1 Encoder

**职责**: 输入编码

**组件**:
- EmbeddingLSTM
- 记忆检索
- 上下文融合

**编码流程**:
```
input_indices → Embedding → LSTM → memory_retrieval → context_fusion → output
```

**记忆检索**:
- 使用编码输出作为查询
- 检索相关记忆
- 融合到上下文

#### 6.2 Decoder

**职责**: 输出生成

**组件**:
- EmbeddingLSTM
- MoE 投影

**生成流程**:
```
h_init, c_init → EmbeddingLSTM → MoE → logits → sampling → output
```

**采样策略**:
- Greedy: argmax(logits)
- Temperature: softmax(logits/T)
- Beam Search: 保留 Top-B 候选

#### 6.3 Nuther Model

**职责**: 完整模型整合

**架构**:
```
Input → Encoder → [LSTM + Memory] → Decoder → [LSTM + MoE] → Output
```

**前向传播模式**:
1. **Teacher Forcing**: 使用目标序列训练
2. **Autoregressive**: 逐步生成

---

### 7. 对话模块 (Chat Module)

#### 7.1 Chat Session

**职责**: 会话管理

**数据结构**:
```python
{
    'session_id': str,
    'start_time': float,
    'message_count': int,
    'dialogue_history': List[Dict]
}
```

**功能**:
- 消息管理
- 历史记录
- 统计信息

#### 7.2 Chat Bot

**职责**: 对话机器人

**功能**:
- 多会话管理
- 交互式对话
- 批处理支持
- 命令处理

**对话流程**:
```
User Input → Encode → Generate with Memory → Decode → Response
```

---

### 8. 主程序 (Main Module)

**职责**: 框架管理

**类**: `NutherFramework`

**功能**:
- 模型初始化
- 模块协调
- 配置管理
- 命令行接口

**管理功能**:
```python
# 初始化
initialize_new()
load(model_path, vocab_path)

# 知识管理
build_vocabulary(texts)
load_knowledge_base(file_path)
crawl_and_store(urls)

# 对话
chat(max_length, temperature)
generate(input_text, max_length, temperature)

# 保存
save(save_dir)
```

---

## 数据流设计

### 输入流程

```
文本输入
  ↓
分词 (tokenize)
  ↓
索引转换 (text_to_indices)
  ↓
嵌入 (Embedding)
  ↓
LSTM 编码
  ↓
记忆检索
  ↓
上下文融合
  ↓
LSTM 解码
  ↓
MoE 投影
  ↓
采样
  ↓
索引转换 (indices_to_text)
  ↓
文本输出
```

### 记忆流程

```
知识文本
  ↓
分块 (chunking)
  ↓
关键词提取
  ↓
存储 (Memory)
  ↓
查询输入
  ↓
相似度计算
  ↓
Top-K 检索
  ↓
上下文融合
```

### MoE 流程

```
输入向量
  ↓
门控网络
  ↓
专家选择 (Top-K)
  ↓
专家计算
  ↓
加权求和
  ↓
输出向量
```

---

## 性能优化

### 1. 计算优化
- 稀疏路由减少计算量
- 批处理提高效率
- 向量化操作

### 2. 内存优化
- LRU 淘汰策略
- 分块存储
- 去重机制

### 3. 检索优化
- 关键词索引
- Top-K 选择
- 阈值过滤

---

## 扩展性设计

### 1. 专家扩展
- 添加新的专家类型
- 自定义门控策略
- 灵活的输出融合

### 2. 记忆扩展
- 支持不同的相似度计算
- 多种检索策略
- 记忆压缩

### 3. 模型扩展
- 支持注意力机制
- Transformer 集成
- 多模态支持

---

## 总结

Nuther 采用分层、模块化的架构设计，各模块职责明确，接口清晰。通过 LSTM、记忆回溯和 MoE 的有机结合，实现了强大的对话能力。架构设计充分考虑了可扩展性、可维护性和性能优化，为后续功能扩展奠定了良好基础。