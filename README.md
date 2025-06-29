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

   （4）总体结构：
    ![image](https://github.com/user-attachments/assets/1b7be113-1792-4c3c-a94a-38701b36ea3e)

3. 构建decorder：
   ![image](https://github.com/user-attachments/assets/e27b5a60-b3b4-4a3b-97eb-adc8051fbdbd)

4. 分割头：
   为什么最后要上采样四倍：decoder结束只恢复到patch_embeded之后的大小，此时还是原图的1/4（准确来讲是1/patch_size，但前面用的patch_size=4）
   ![image](https://github.com/user-attachments/assets/d55f6714-a056-4b97-86ca-9558a2a2d2bf)




   
   
