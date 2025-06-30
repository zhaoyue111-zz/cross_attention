# cross_attention
前提：共5个域
| 域编号 | 训练集数量 | 测试集数量 | 数据来源               |
|------|------------|------------|------------------------|
| 01   | 51         | 50         | DRISHTI-GS    |
| 02   | 99         | 60         | RIM-ONE-r3  |
| 03   | 320        | 80         | REFUGE-train           |
| 04   | 320        | 80         | REFUGE-val             |
| 05   | 320        | 80         | REFUGE-test            |


整体思路：
1. 构建输入图像对：这里需要确定一个目标域，将目标域的图片按顺序与源域的图片随机组合，沿通道方向堆叠，目标域图片在前。比如要分割的是1域，那么就组合(1_1, 2_5),(1_1, 3_78),(1_1, 4_55),(1_1, 5_67),(1_2, 2_78)....其中来自1域的图片形成前3个通道，来自其它域的图片形成后3个通道。代码见mydataset.py，其中包括数据增强。
2. 构建 encoder（`swintransformerv2.py`）：

   - （1）Patch Embedding: `(B, H, W, 3)` → `(B, L, C)`
     
     ```python
     core: x = self.proj(x).flatten(2).transpose(1, 2)
     ```

   - （2）加位置编码：
     
     - **绝对位置编码**：
       
       ```python
       self.absolute_pos_embed = nn.Parameter(
           torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
       )
       trunc_normal_(self.absolute_pos_embed, std=.02)
       ```

     - **相对位置编码**：
       
       ```python
       self.pos_embd = SinPositionalEncoding2D(embed_dim).cuda()
       ```

   - （3）通过 4 (`num_layers`) 层的 `BasicLayer`：
     
     - 每个 `BasicLayer` 包含 4 层（`depths`）Swin Transformer 和一层 `PatchMerging`
     - 每经过一层：
       - 通道数 `C` → 2 倍
       - Token 数 `L` → 1/4，即 `(H×W) → (H/2×W/2) = H×W/4`

    ![image](https://github.com/user-attachments/assets/4327ea3a-fd4a-40e4-9e08-445d55d5d07f)

   这里应考虑depths的设置，传统的swint如图：
   ![image](https://github.com/user-attachments/assets/bf8e0b99-e2a4-4987-aa5c-8405c16af84c)
   较深的层理论上能获取到更深层次的语义信息，但同时应考虑可能出现的梯度消失/爆炸和过拟合问题。

   （4）总体结构：
    ![image](https://github.com/user-attachments/assets/1b7be113-1792-4c3c-a94a-38701b36ea3e)

4. 构建decorder：
   ![image](https://github.com/user-attachments/assets/e27b5a60-b3b4-4a3b-97eb-adc8051fbdbd)

5. 分割头：
   为什么最后要上采样四倍：decoder结束只恢复到patch_embeded之后的大小，此时还是原图的1/4（准确来讲是1/patch_size，但前面用的patch_size=4）
   ![image](https://github.com/user-attachments/assets/d55f6714-a056-4b97-86ca-9558a2a2d2bf)

6. memory bank：
   使用了简单的列表存储特征，给每个域分配一个bank。更新域中保存的特征在最终交叉分割和单独分割都得到结果后进行，如果交叉分割的损失比单独分割的损失小，且大于一个最小阈值就将其保存到对应域的bank中；如果当前保存的特征数=max_size，就将保存的最差得分对应的特征替换为当前的特征（因此需要计算每个特征的得分-->如果对某个类别表现出明显的倾向性，就认为其得分高；如果对每个类别都“差不多”，得分就低），基于域的损失有两项：域间损失和域内损失，域内损失仅在max_size>1时使用（也就是每个域至少保存top2个特征），域间损失使用被分割的目标域为正样本，其他所有域均为负样本。域间损失应最大化，域内损失应最小化。最终得到的对比损失是平均域内损失和平均域间损失的和。

   但是我是自己实现的损失计算方法，细节处理肯定不如官方的细致，可以使用标准的库函数pml_losses.NTXentLoss。如果是基于自己实现的函数，除了考虑代码实现层面的细节（比如计算相似度的方法），还应该考虑对比损失中两项的权重，memory bank的max_size大小，衡量损失的方法（怎么表示域间最大域内最小），memory bank更新的方法是否激进（比如是否使用动量memory bank）等问题。
   

训练细节：
1. 关于encoder通过每层basic layer后的通道维度：
   
     ```python
           if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
           elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
           elif size.split("_")[1] in ["large"]:
            feature_channels = [384, 768, 1536, 1536]
     ```
这个可以个人指定，暂未在不同的通道数上做消融。

2. 损失项只考虑了：单独分割和真实掩码之间的损失、交叉分割和单独分割之间的损失、memory bank对比损失，可以考虑其他损失，比如encoder提取特征的稳定性（语义信息应一致，kl divergence），交叉分割与真实掩码之间的损失。另外可以考虑不同损失项之间的权重。
   
