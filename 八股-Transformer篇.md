![image-20250316235105908](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20250316235105908.png)

### 1. 注意力机制和自注意力机制有什么区别？

**传统注意力机制**：由于传统的 **Encoder-Decoder** 架构在建模过程中，下一时刻的计算过程会依赖于上一个时刻的输出，即整个过程需要按序进行，而这种固有的属性就限制了模型不能以并行的方式进行计算。

**自注意力机制**：无时间依赖性，可并行计算。在 **Transformer** 中，输入序列的所有 **token** 可以同时计算 **Query、Key、Value** 并进行注意力计算，从而避免了逐步递归的限制。同时，每个 **token** 可以直接关注整个序列，长距离依赖效果更好。



### 2. 自注意力机制公式？

$$
\text{Attention}(Q, K, V) = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

主要计算单词之间的相互影响。



### 3. 介绍一下Attention的全流程

Attention 机制是 Transformer 模型的核心，它允许模型在处理序列数据时动态关注不同位置的 Token，提高模型的全局感知能力。

**Step 1：输入数据**

- 批量大小（Batch Size）： $B$

- 序列长度（Sequence Length）： $L$

- 嵌入维度（Embedding Dimension）：$d_{model}$

- 注意力头数（Number of Heads）： $h$

- 单个头的维度（Head Dimension）： $d_k$（一般 $d_k = d_{model}/h$）

输入的 token 为：$ X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$

可训练的权重矩阵有：
$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$
其中，$W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times h \times d_k}$

投影后 Query, Key, Value 形状为：
$$
Q, K, V \in \mathbb{R}^{B \times h \times L \times d_k}
$$
**Step 2：计算相似性得分**

计算查询（Query）和键（Key）之间的点积：
$$
S = \frac{Q K^T}{\sqrt{d_k}}，~~~~S \in \mathbb{R}^{B \times h \times L_Q \times L_K}
$$
**Step 3：应用 Mask（decoder中）**
$$
S = S + \text{mask}
$$
Mask 形状：$(B, 1, L_Q, L_K)$

填充值：1e-9（理论）

**Step 4：计算注意力权重（Softmax 归一化）**

对相似性得分 $S$ 进行 Softmax 归一化：
$$
A = \text{softmax}(S),~~~A \in \mathbb{R}^{B \times h \times L_Q \times L_K}
$$
**Step 5：计算加权求和得到 Attention 输出**

用注意力权重 $A$ 乘以值矩阵 $V$ :
$$
O = A V,~~~~O \in \mathbb{R}^{B \times h \times L \times d_K}
$$
**Step 6：组合多头输出（Concatenation）**

将所有 `h` 个头的输出拼接，然后通过一个线性变换  $W^O$ 投影回 `d_model` 维度：
$$
O_{\text{final}} = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$
拼接后形状：$O_{\text{concat}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$

最终输出形状：$O_{\text{final}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$



### 4. 为什么要除以 $\sqrt{d_k}$？

在多头自注意力中，$d_k$ 指的是 **key** 的维度，通常等于总模型的维度 $d_{model}$ 除以多头的数量。

数值级对 **softmax** 得到的分布影响非常大，之所以要除以 $\sqrt{d_k}$，是因为当 $d_k$ 增大时，点积的值会变得很大，容易导致 **softmax** 产生梯度消失的问题，影响反向传播的结果。

#### softmax 的梯度：

$$
\frac{\partial A_i}{\partial S_j} =
\begin{cases}
A_i(1 - A_i), & i = j \\

- A_i A_j, & i \neq j
  \end{cases}
$$

可以看出：

- **当 softmax 输入 $S_i$ 数值较小且接近 0 时，梯度最大。**
- **当 $S_i$ 数值较大（过大或过小），softmax 的梯度变得很小，会导致梯度消失问题。**

$Q$ 和 $K$ 的每个元素都是随机初始化的，均值大约为 0，方差接近 1，因此：
$$
\text{Var}(QK^T) \approx d_k
$$
点积的方差大约是 $d_k$，采用 $\sqrt{d_k}$ 作为缩放因子，可以在数值稳定性和梯度流动之间取得平衡。

因此，对于 $q \cdot k$，其均值为：
$$
E(q \cdot k) = 0
$$
方差为：
$$
D(q \cdot k) = d_k
$$
方差越大也就说明，点积的数值级越大（以越大的概率取极值）。一个自然的做法就是把方差稳定到 1，做法是将点积除以 $\sqrt{d_k}$，这样有：
$$
D\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = 1
$$
将方差控制为 1，也就有效地控制了前面提到的梯度消失的问题。



### 5. softmax 的梯度计算与梯度消失问题

$$
\frac{\partial g(x)}{\partial x} = \text{diag}(\hat{y}) - \hat{y} \hat{y}^T \quad \in \mathbb{R}^{d \times d}
$$

把这个矩阵展开：
$$
\frac{\partial g(x)}{\partial x} =
\begin{bmatrix}
\hat{y}_1 & 0 & \cdots & 0 \\
0 & \hat{y}_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \hat{y}_d
\end{bmatrix}
-
\begin{bmatrix}
\hat{y}_1^2 & \hat{y}_1 \hat{y}_2 & \cdots & \hat{y}_1 \hat{y}_d \\
\hat{y}_2 \hat{y}_1 & \hat{y}_2^2 & \cdots & \hat{y}_2 \hat{y}_d \\
\vdots & \vdots & \ddots & \vdots \\
\hat{y}_d \hat{y}_1 & \hat{y}_d \hat{y}_2 & \cdots & \hat{y}_d^2
\end{bmatrix}
$$

根据前面的讨论，当输入 **x** 的元素较大时，softmax 会把大部分概率分配给最大的元素。

假设我们的输入数值极大，最大的元素是 $x_1$，那么 softmax 计算出的 $\hat{y}$ 就将产生一个接近 **one-hot** 的向量：
$$
\hat{y} \approx [1, 0, \cdots, 0]^T
$$

此时上面的矩阵变为如下形式：
$$
\frac{\partial g(x)}{\partial x} \approx
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
= 0
$$

也就是说，在输入的数值极大时，**梯度消失** 为 0，造成参数更新困难。



### 6. 什么softmax的数值上溢和数值下溢？在实际中如何处理？

数值上溢（Overflow）：当 Softmax 输入的数值过大时，计算指数时会导致溢出。

数值下溢（Underflow）：当 Softmax 输入的数值过小时，计算指数时会导致接近零的结果。

在实际取 bf16 的情况下，有效范围是 $(-126,127)$.

**处理方法**：

- **减去最大值**：通过对输入向量进行处理，每个元素减去这个向量的最大的元素，不会改变最终的结果。

$$
\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}
$$

- **log-sum-exp 技巧**：对于涉及指数和求和的表达式，可以将其转化为对数形式，从而避免直接计算大的指数值。（在这里最终结果与上面的方法相同）

$$
\log \left( \sum_{j} e^{x_j} \right) = \max_j(x_j) + \log \left( \sum_{j} e^{x_j - \max_j(x)} \right)
$$

- **使用温度（Temperature）**：使用温度可以一定程度上解决 Softmax 中的溢出和下溢问题。温度较小时，Softmax 输出的分布会更加极端，主要依赖于最大值。这会导致溢出的可能性增加，特别是如果输入值非常大时。温度较大时，Softmax 输出的概率分布会趋于均匀，减小了极端数值对输出的影响。这种情况下，由于指数函数的平滑，数值溢出和下溢的风险减少。

$$
\text{Softmax}(x_i, T) = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}}
$$



### 7. 为什么要分为多头？

单头注意力只能学习一种注意力模式，容易丢失信息。多头保证了 **Transformer** 可以注意到不同子空间的信息，捕捉到更加丰富的特征信息，增强对输入序列的表达能力。同时，多头注意力可以进行 **并行** 计算，适合 **GPU** 的计算方式，可以提升计算效率。



### 8. 为什么 **Q** 和 **K** 使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？

**K** 和 **Q** 的点乘是为了计算一个句子中每个 **token** 相对于句子中其他 **token** 的相似度，得到一个 **attention score** 矩阵。然而，**V** 还代表着原来的句子，所以我们拿这个 **attention score** 矩阵与 **V** 相乘，得到的是一个加权后结果。

**K** 和 **Q** 使用了不同的 **W_k**、**W_q** 来计算，可以理解为是在不同子空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的 **attention score** 矩阵的泛化能力才更强。

但是如果不使用 **Q**，直接拿 **K** 和 **K** 点乘的话，你会发现 **attention score** 矩阵是一个对称矩阵。因为是同样一个矩阵，都会投影到了同样一个空间，所以泛化能力很差。这样的矩阵导致 **V** 进行加权时，效果也不会好。



### 9. Transformer中编码器和解码器的作用是什么？

**Encoder**：理解输入序列，通过自注意力机制捕捉输入序列的全局依赖关系，并生成富含语义信息的上下文表示（Hidden States）。注意，Encoder 经过所有层处理后才会与 Decoder 交互（论文中是N=6）。

**Decoder**：生成输出序列，通过自注意力机制建模已生成的部分序列。通过交叉注意力机制结合编码器的输出，以确保生成的结果与输入匹配，最终逐步生成最终的输出序列。



### 10. Decoder 中 Mask（Causal Mask）的作用

**训练阶段**：输入完整句子，但 Mask 强制遮挡未来 token，确保模型只能利用已生成的部分来预测下一个 token。Masked Self-Attention 通过上三角矩阵 Mask实现：
$$
\begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$
**推理阶段**：推理时，每次只输入已生成的 token，所以天然符合 Mask 规则。当前 Query 只与已有的 token 计算注意力。

|          **场景**          | **Query 形状（$W_Q$）** |       **Key 形状（W_K）**       |         **Mask 形状**          |
| :------------------------: | :---------------------: | :-----------------------------: | :----------------------------: |
|  **训练时（全序列计算）**  |    $(B, h, L, d_k)$     |        $(B, h, L, d_k)$         |         $(B, 1, L, L)$         |
| **LLM 生成时（单步计算）** |    $(B, h, 1, d_k)$     | ($B, h, L_{\text{cache}}, d_k)$ | $(B, 1, 1, L_{\text{cache}}) $ |



### 11. 在计算attention score的时候如何对padding做mask操作（Padding Mask）？

在计算 Attention Score 时，对 padding 进行 mask 处理的目的是避免模型对填充部分进行关注，以确保注意力机制仅专注于实际的有效序列信息。
$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + \text{mask} \right) V
$$
Padding Mask 的填充值应该设为一个极小值（-1000），这样 Softmax 分布中的该位置概率几乎变为 0，不会影响最终计算。

Query 不能访问 Key 的 Padding Token，最终形状依旧是 $(B,1,1,L_{cache}).$

Causal Mask 可以和 Padding Mask 同时使用：`final_mask = causal_mask & padding_mask`
$$
\text{Final Mask}=\text{Causal Mask}∧\text{Padding Mask}
$$



### 12. 为什么多头注意力需要降维？

首先是为了降低计算复杂度，避免计算量随着 h 线性增加。同时，使用多头让不同头关注不同信息，提升 Transformer 表达能力。这也符合 Transformer 结构设计，最终的 Concat + $W^O$ 变换回 $d_{\text{model}}$ 维度。

总之，如果不降维，计算量会成倍增长，模型训练成本变高，且不同头学到的特征可能是冗余的。降维后，多头可以更有效地学习不同的注意力模式，增强模型的表达能力。



### 13. 大概介绍一下Transformer的Encoder模块

一个完整的 Transformer Encoder 由多个相同的 Encoder Layer 叠加而成，每个 Encoder Layer 主要包含：

- 输入嵌入（首次Input Embedding + Positional Encoding，后续传入上一个encoder的输出） 

- 多头自注意力（Multi-Head Self-Attention, MHA） 

- 残差连接 + 层归一化（Residual Connection + LayerNorm）

- 前馈神经网络（Feed-Forward Network, FFN） 

- 残差连接 + 层归一化（Residual Connection + LayerNorm）
  $$
  \text{Encoder Output} = \text{LayerNorm}(\text{FFN} + \text{LayerNorm}(\text{MHA} + X))
  $$



### 14. 为什么要先残差连接再归一化，有什么优势？

归一化后的数据会改变原始数据的分布，使得梯度更新可能受到影响。如果先残差再归一化，梯度可以更好地传播，不容易消失或爆炸，确保残差连接能有效保留输入信息。

如果先归一化再残差，梯度可能会受到 LayerNorm 的影响，使得训练变得更不稳定。不过它能使得 Transformer 在更深层时更稳定。

**残差连接的作用：**

- 解决梯度消失问题，由于梯度反向传播时不断乘以链式法则中的权重矩阵，导致梯度逐渐消失，最终使得网络很难进行有效的训练。通过引入残差连接，网络的梯度可以在反向传播时通过直接的跳跃连接传递，从而缓解了梯度消失的问题。这使得深层网络在训练时更加稳定，并能有效地训练更深的网络。
- 信息流动更为顺畅，信息通过跳跃连接可以直接到达较浅的层，避免了信息的丢失。
- 残差连接本质上提供了一种恒等映射的学习方式。如果某一层的变换效果不好，网络可以通过学习一个零变换来“跳过”该层的变换。
- 加速训练并改善性能，网络可以更加灵活地学习到更复杂的表示，模型可以学习到更多的特征，并且在相同数量的训练迭代中能够更快地收敛。



### 15. FFN 的结构是怎样的？有什么作用？

**结构**：FFN 是由两层全连接层（Linear Layers）+ 非线性激活函数（ReLU / GELU）组成：
$$
\text{FFN}(X) = \text{ReLU}(X W_1 + b_1) W_2 + b_2
$$

- 输入 Token 表示：$X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- 第一层线性变换 + 非线性激活：$X' = \text{ReLU}(X W_1 + b_1) \quad \text{(维度: } d_{\text{model}} \to d_{\text{ff}}:4*d_{\text{model}} \text{)}$
- 第二层线性变换：$X_{\text{FFN Output}} = X' W_2 + b_2 \quad \text{(维度: } d_{\text{ff}} \to d_{\text{model}} \text{)}$

**作用**：引入非线性变换，增强特征表达能力，独立作用于每个 Token，类似于 CNN 中的 1x1 卷积，起到特征提取作用。保持 `d_model` 维度不变，使得多个 Encoder 层可以顺利堆叠。

**变种**：
$$
\text{GELU}(x) = x \cdot \Phi(x)
$$
其中 $\Phi(x)$ 是标准正态分布的 CDF。

GELU 更平滑，能减少 ReLU 造成的“死亡神经元”问题；ReLU 更简单，计算开销比 GELU 低。



### 16. 大概介绍一下Transformer的Decoder模块

**Decoder 的输入是**：目标序列（模型希望生成的输出序列，如翻译任务中的目标语言文本；在 decoder 之间传递的是新 token 的隐藏状态）；Encoder 提供的上下文信息（即源序列经过编码器后的表示）。

**Decoder 的输出是**：预测的目标序列（如翻译后的文本）；在训练过程中，Decoder 通过 **teacher forcing** 机制使用真实的目标序列作为输入，而在推理过程中，它会根据自己已生成的部分逐步预测下一个token。

经典 Transformer 中包含下面三个关键结构，在每个结构后面通过 **Add ＆ Norm** 连接。

- **Masked Multi-Head Self-Attention（掩码多头自注意力）**：计算目标序列内部的注意力，同时屏蔽未来的 tokens（防止信息泄露。$Q$,$K$,$V$ 由目标序列的嵌入得到。
- **Multi-Head Cross-Attention（交叉多头注意力）**:让 Decoder 关注 Encoder 生成的上下文信息。$K$ 和 $V $由 Encoder 提供，而 $Q$ 由 Decoder 提供。更准确的说，权重矩阵依旧是当前 Attention 的权重矩阵，不过权重矩阵作用的序列（隐藏状态）分别来自于 Encoder 的最终输出以及掩码多头自注意力的输出。
- **Feed-Forward Network（前馈全连接网络）**：每个 token 独立变化。

在所有的 decoder 结束后，输出层还有两层额外的结构：

- **线性变换（Linear Layer）**：Decoder 最后一个隐藏状态 $h_t$ 先经过一个线性变换。将隐藏状态 $h_t$ 转换为词汇表大小的 logits，这些 logits 代表每个单词的分数，表示模型认为它是正确词的可能性。

$$
z_t = W_o h_t
$$

$W_O \in (d_{model},V)$ 是可训练矩阵，$V$是词汇表大小。训练时，$h_t \in (B,L,d_{model})$；推理时，$h_t \in (B,1,d_{model})$. $z_t$ 的形状就是将 $h_t$ 的 $d_{model}$ 转变为 $V$.

- **Softmax 计算概率分布**：对 logits $z_t$ 进行 Softmax 归一化，得到概率分布：

$$
P(y_t | y_{<t}, x) = \text{Softmax}(z_t) = \frac{e^{z_t}}{\sum_{j=1}^{V} e^{z_j}}
$$

最终选择最大概率的词作为 $y_t$：$y_t = \arg\max P(y_t | y_{<t}, x)$.



### 17. 为什么编码器的输入叫做 input，而解码器的输入叫做 target？

解码器的输入有些特殊，这个 **"target"** 其实是目标序列的一部分，有时候会叫做 **"output"**，也就是说解码器的输入和输出都是 output，具体来说如下：

**训练阶段**：一种 **教师强制（Teacher Forcing）** 的训练方法，在 **Seq2Seq** 中很常见，使用 **ground truth** 作为输入。也就是说，把 **整个目标序列作为输入**（训练过程更加稳定和高效，因为模型可以一次性看到所有正确的上下文信息）。在这个过程中，使用了 **future mask（Mask Attention）** 来确保在预测每个词时，模型只能使用该词之前的词，防止信息泄露。

**推理（测试）阶段**：把上一个时刻 Decoder 的结果当作当前时刻的输入。这种方法也常叫做 **自回归生成（Auto-Regressive Generation）**，从一个特殊的 开始标记（例如 `<start>`）开始，模型逐步生成序列。每一步生成的词将被添加到已生成的序列中，并用作下一步的输入。逐步解码，在每一步，模型基于**已生成的序列**（加上编码器的输出）来预测下一个 token。这个过程会一直持续到模型生成一个 **特殊的结束标记**（例如 `<end>`）为止。



### 18. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？

Transformer 本身没有循环（RNN）或卷积（CNN）结构，意味着它无法直接捕捉序列的位置信息。在 Self-Attention 机制 中，输入序列的所有 token 彼此平等，没有先后顺序。
$$
PE(pos, 2i) = \sin \left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)
$$

$$
PE(pos, 2i+1) = \cos \left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)
$$

**正余弦固定位置编码**：

- pos 是 token 在序列中的位置索引（0, 1, 2, ...）；
- i 是隐藏维度的索引（偶数维用正弦，奇数维用余弦）；
- $d_{model}$ 是 Transformer 的隐藏层维度（如 512）；
- 10000 是一个缩放因子，用于调整不同维度上的周期性。

**优点**：

- **无参数（Fixed）**：不需要学习额外的参数；
- **平滑（可微）**：不会像索引一样带来梯度问题；
- 可以扩展到任意长度。

**缺点**：

- 固定编码缺乏灵活性；
- 只能表示绝对位置关系，经过线性变化相对位置信息丢失。



### 19.  介绍一下相对位置编码（Relative Positional Encoding, RPE）

Attention 计算的是 token 之间的关系，并不直接需要绝对位置，而是需要相对位置信息（即 token 之间的距离）。
$$
\text{Attention}(Q, K, V, R) = \text{softmax} \left( \frac{QK^T + Q R^T + S}{\sqrt{d_k}} \right) V
$$
其中，$R$ 是相对位置编码矩阵，表示不同 token 之间的相对距离信息，$S$ 额外的偏置项，可用于调节注意力分布。

还有一种实现的方式是，引入可学习的相对位置编码 $a_{ij}^K$ 和 $a_{ij}^V$，在 Attention 计算的时候，将 $K_j = x_jW^K$ 变换为 $K_j = x_j W^K + a_{ij}^K$，i 表示第i个 Query。同理替换 Value。



### 20. 介绍一下旋转位置编码（Rotary Positional Embedding, RoPE）

核心思想：把每个 token 的隐藏向量的每一对维度（偶数-奇数）进行 **二维旋转**，旋转角度由 token 的位置和一组预定义频率决定。

RoPE是 一种特殊的相对位置编码方法，通过 **旋转矩阵** 在嵌入空间中编码相对位置信息，使得 Self-Attention 能够隐式感知 token 之间的相对关系。传统的位置编码是将位置信息直接加到 token 的嵌入中，而 RoPE 是将位置编码转换为旋转矩阵作用于 Query 和 Key（注意：不作用于 Value）。通过旋转角度编码相对位置，使得 Attention 计算时，位置信息自然体现在点积运算中。

对于 $d$ 维的 Query 向量 $q$，每两维作为一对，执行二维旋转变换：
$$
\text{RoPE}(q, pos) =
\begin{bmatrix}
q_1 \cos(pos*\theta) - q_2 \sin(pos*\theta) \\
q_2 \cos(pos*\theta) + q_1 \sin(pos*\theta) \\
\vdots
\end{bmatrix}=R_{pos}q
$$
其中，旋转角度 $\theta$ （也称作逆频矩阵）的计算方式：
$$
\theta = 10000^{-2(i-1)/d}
$$

- $pos$：当前 token 在序列中的位置索引。
- $i$：隐藏向量的维度索引（每两维一组，$i = 1$ 表示 $q_1$ 和 $q_2$ 两个元素的位置）,$i \in [1,2,..,d/2]$。
- $d$ ：隐藏层维度。

两两一组旋转的公式如下：
$$
\begin{bmatrix} q_i' \\ q_{i+1}' \end{bmatrix} =
\begin{bmatrix} 
\cos(\theta) & -\sin(\theta) \\ 
\sin(\theta) & \cos(\theta) 
\end{bmatrix}
\begin{bmatrix} q_i \\ q_{i+1} \end{bmatrix}
$$
低维部分（$i$ 小的维度）变化较快，编码较局部的信息。

高维部分（$i$ 大的维度）变化较慢，编码较全局的信息。

对于位置 $M$ 和 $N$，RoPE通过旋转矩阵 $R_M$ 和 $R_N$ 分别对查询向量 q 和键向量 k 进行变换：
$$
(R_Mq)^T(R_Nk) = q^TR_{N-M}k=q^TR^T_{M-N}k
$$
**优点**：

- **相对位置感知**，通过旋转操作，使得 Attention 隐式建模相对位置信息；
- **无需额外参数**，不需要额外的可学习参数，计算更高效；
- **适应长序列**，RoPE 可以无缝扩展到比训练时更长的序列；
- **适用于所有 Transformer 结构**，兼容 GPT、BERT、T5 等 Transformer 模型。



### 21. 为什么 RoPE 只作用于 $Q$ 和 $K$，而不作用于 $V$？

 RoPE 影响的是注意力计算，而不是 Value 信息。RoPE 的目的是在 $QK^T$ 计算中引入相对位置信息，使 Attention 机制能够捕捉相对位置依赖，明确哪些位置的Token应该彼此关注。$V$ 只用于信息聚合，它携带的是 token 的语义信息，不参与注意力计算本身，因此 RoPE 不需要影响 $V$。

实验也表明，对 $V$ 施加位置编码的变体，生成的效果下降，同时引入了额外的噪声，影响梯度传播。



### 22. 为什么RoPE能处理长序列？

外推能力指模型在训练时未见过的长序列（如长度超过训练时的最大上下文窗口）上的表现。RoPE的外推优势源于其旋转编码的**线性性**和**频率分布设计**。

 **旋转的线性叠加性**：$R_{m+n}=R_m R_n$

**频率的几何级数分布**：RoPE的旋转角 $\theta_i = 10000^{-2i/d}$ 按几何级数分布，

- **长波长**（低频子空间）：覆盖大范围位置差异（如 $\theta_i$ 较小，$cos⁡(mθ_i)$ 变化缓慢）。
- **短波长**（高频子空间）：捕捉局部位置关系（如 $\theta_i$ 较大，$cos⁡(mθ_i)$ 变化剧烈）。

这种设计使得模型在训练时已学习到多种尺度的位置模式，面对更长的序列时，低频子空间仍能有效表征远距离位置关系。



### 23. RoPE 怎么作用于 KV 缓存

新增第n+1个token 时，其位置为 $pos=n+1$.计算当前 token 旋转后的表示：
$$
q^{rot}_{n+1} = R_{n+1} \cdot q_{n+1}
$$
此时，缓存的 k 已预先应用了对应位置的旋转矩阵：
$$
k_j^{rot} = R_j \cdot k_j  ~~~~(1 <= j <= n)
$$
接下来计算注意力得分：
$$
AttentionScore(q_{n+1},k_j)=(q^{rot}_{n+1})^T \cdot k_j^{rot}=q^T_{n+1}R^T_{n+1-j}k_j
$$



### 24. Transformer 训练时的学习率设定

在 Transformer 模型的训练过程中，学习率（learning rate, LR）的设定对模型的收敛速度和最终性能至关重要。一般采用 **学习率调度（learning rate scheduling）** 来动态调整学习率，以提高训练效果。

常见的 Transformer 训练学习率调度方式包括：

-  **预热（Warm-up）+ 逐步衰减（Decay）**

论文 *Attention Is All You Need*中，学习率设定为：

$$
lr = d_{\text{model}}^{-0.5} \times \min(step^{-0.5}, step \times \text{warmup\_steps}^{-1.5})
$$
其中：

- $d_{\text{model}}$ 是 Transformer 的隐藏层维度（一般为 512 或 1024）。
-  $step$  是当前的训练步数。
-  $\text{warmup\_steps}$ 是预热步数（如 4000）。

作用：在训练初期使用较低的学习率，逐渐增加至峰值，然后缓慢下降，以保证模型稳定收敛。

- **余弦退火（Cosine Annealing）**

采用余弦退火策略，从初始学习率缓慢衰减：
$$
lr_t = lr_{\max} \times 0.5 \times \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)
$$
其中：$ t$  为当前 epoch, $T$ 为最大 epoch。

- **阶梯衰减（Step Decay）**

每隔固定步数（如 10k steps 或若干 epochs）将学习率缩小一半。

- **OneCycleLR**

适用于大批量训练，先快速升高学习率，再慢速降低到原始学习率的 1/10 或更低。

**预训练阶段**：学习率较小，例如 **1e-4** 或 **5e-4**。采用 **Warm-up + Decay**（如 warm-up 4000 steps）。

**微调阶段**：学习率更小，如 **1e-5** 到 **5e-5**。使用 **线性衰减（Linear Decay）** 或 **余弦衰减（Cosine Decay）**。



### 25. Dropout是如何设定的，位置在哪里？

- **注意力层的注意力权重（Attention Weights Dropout）**：在计算 $Softmax(\frac{QK^T}{\sqrt{d_{\text{model}}}})$ 之后，对注意力分数进行 Dropout
  $$
  \text{Dropout}(\text{ReLU}(W_1 X + b_1)) W_2 + b_2
  $$
  
- **FFN（前馈网络）**: Dropout 作用在 ReLU 或 GELU 之后
  $$
  \text{Dropout}(\text{ReLU}(W_1 X + b_1)) W_2 + b_2
  $$

- **Transformer 论文默认值**：0.1 Embedding 层的 Dropout：在 输入 token embedding + 位置编码（Positional Encoding）之后进行 Dropout
  $$
  \text{Dropout}(\text{Embedding}(X) + \text{PositionalEncoding})
  $$

- **残差连接后**：在 Add & Norm 之前，对残差连接（Residual Connection）进行 Dropout
  $$
  X + \text{Dropout}(\text{LayerNorm}(Y))
  $$

Dropout 率设定：Transformer 论文默认值：0.1，小模型可以增加（0.2），大模型可以减小（0.05）。

推理时记得关闭 dropout。Dropout 在训练时会随机丢弃神经元，并对激活值进行缩放（如 `p=0.1` 时，缩放因子为 `1/(1-0.1)=1.11`）。在推理时必须关闭 Dropout，否则模型输出会不稳定。



### 26. Transformer的并行化提现在哪个地方？

- **自注意力机制（Self-Attention）的并行化**：矩阵计算可以并行化，可以并行计算所有 token 的注意力分数。同时，由于每个 head 计算是独立的，因此可以完全并行计算多个头的注意力，最后拼接。

- **前馈网络（FFN）的并行化**：由于 FFN 作用在每个 token上，并且所有 token 共享相同的权重，因此它可以完全并行化计算。

- **残差连接与层归一化**：LayerNorm 作用于单个 token 的表示，不受序列长度影响，可以独立计算。残差连接仅是矩阵加法，计算量极小，也不会影响并行性。

- **Decoder 自注意力（Masked Self-Attention）的限制**：在训练时，整个目标序列是已知的，因此 Masked Self-Attention 仍然可以并行计算。训练时的 Transformer Decoder 与 Encoder 端并行程度几乎相同。

  在**推理**（生成文本）时，由于 Transformer 需要 逐步生成 token：第一个 token 生成后，必须等到它经过 Softmax 选择出具体的 token，再将其作为输入，生成下一个 token。由于这种 序列依赖性（sequential dependency），推理时 Decoder 端无法完全并行化，只能按 token 逐步进行。
