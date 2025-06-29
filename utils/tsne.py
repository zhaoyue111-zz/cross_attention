from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_domain_from_path(img_path):
    """从图像路径中提取域信息
    Args:
        img_path: 图像路径，可能是tensor、numpy数组或字符串
    Returns:
        domain_id: 域ID (0-4) 或 None
    """
    try:
        # 添加调试信息
        print(f"[DEBUG] 处理图像路径: {img_path}")
        
        if not isinstance(img_path, str):
            if isinstance(img_path, (torch.Tensor, np.ndarray)):
                img_path = str(img_path)
            else:
                print(f"[WARNING] 未知的路径类型: {type(img_path)}")
                return None

        # 从路径中提取域号
        parts = img_path.split('/')
        print(f"[DEBUG] 路径分割结果: {parts}")  # 添加调试信息
        
        for part in parts:
            if part.strip() in ['01', '02', '03', '04', '05']:
                domain_id = int(part.strip()) - 1  # 将1-5映射到0-4
                print(f"[DEBUG] 找到域ID: {domain_id}")  # 添加调试信息
                return domain_id

    except Exception as e:
        print(f"[ERROR] 域ID提取失败: {str(e)}")
        print(f"[ERROR] 问题路径: {img_path}")
        return None

    print(f"[WARNING] 无法从路径提取域信息: {img_path}")
    return None

def visualize_sample_features(model, dataloader, save_path, device):
    """使用t-SNE可视化每个样本的KV和Q特征"""
    features_list_k = []  # K特征
    features_list_v = []  # V特征
    features_list_q = []  # Q特征
    domains_list = []
    
    print("[INFO] Starting feature extraction...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, label1, domain_idx) in enumerate(dataloader):
            try:
                for i, d_idx in enumerate(domain_idx):
                    # 获取当前样本的图像
                    img = img1[i:i+1].to(device)  # 使用img1
                    if img.shape[-1] == 3:  # 如果通道在最后
                        img = img.permute(0, 3, 1, 2)
                    
                    # 提取特征
                    q_features, kv_features = model.encoder.extract_domain_kv(img)
                    
                    # 在处理特征之前添加检查
                    if q_features is None or kv_features is None:
                        print(f"[WARNING] Invalid features in batch {batch_idx}, sample {i}")
                        continue

                    # 检查特征维度
                    if q_features.numel() == 0 or kv_features[0].numel() == 0 or kv_features[1].numel() == 0:
                        print(f"[WARNING] Empty features in batch {batch_idx}, sample {i}")
                        continue
                    
                    # 处理K特征
                    key_features = kv_features[0][i:i+1]
                    B, num_heads, L, head_dim = key_features.shape
                    key_features = key_features.transpose(1, 2).reshape(B, L, -1).mean(dim=1)
                    
                    # 处理V特征
                    value_features = kv_features[1][i:i+1]
                    value_features = value_features.transpose(1, 2).reshape(B, L, -1).mean(dim=1)
                    
                    # 处理Q特征
                    query_features = q_features[i:i+1]
                    query_features = query_features.transpose(1, 2).reshape(B, L, -1).mean(dim=1)
                    
                    # 添加到列表
                    features_list_k.append(key_features.cpu())
                    features_list_v.append(value_features.cpu())
                    features_list_q.append(query_features.cpu())
                    domains_list.append(d_idx.item())
                
                if batch_idx % 10 == 0:
                    print(f"[INFO] Processed batch {batch_idx}/{len(dataloader)}")
                    
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {str(e)}")
                continue
    
    if not features_list_k:
        raise RuntimeError("No features were collected!")
    
    # 堆叠所有特征
    features_k = torch.cat(features_list_k, dim=0).numpy()
    features_v = torch.cat(features_list_v, dim=0).numpy()
    features_q = torch.cat(features_list_q, dim=0).numpy()
    domains = np.array(domains_list)
    
    print(f"[INFO] Collected features from {len(np.unique(domains))} domains")
    for d in np.unique(domains):
        print(f"Domain {d}: {np.sum(domains == d)} samples")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # 创建图形 - 现在是2x2的布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    
    # 颜色和标记设置
    colors = ['r', 'g', 'b', 'c', 'm']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 1. Key特征
    k_tsne = tsne.fit_transform(features_k)
    for i in range(5):
        mask = domains == i
        if np.any(mask):
            ax1.scatter(k_tsne[mask, 0], k_tsne[mask, 1],
                       c=colors[i], marker=markers[i],
                       label=f'Domain {i}', alpha=0.6)
    ax1.set_title('Key Features t-SNE')
    ax1.legend()
    
    # 2. Value特征
    v_tsne = tsne.fit_transform(features_v)
    for i in range(5):
        mask = domains == i
        if np.any(mask):
            ax2.scatter(v_tsne[mask, 0], v_tsne[mask, 1],
                       c=colors[i], marker=markers[i],
                       label=f'Domain {i}', alpha=0.6)
    ax2.set_title('Value Features t-SNE')
    ax2.legend()
    
    # 3. Query特征
    q_tsne = tsne.fit_transform(features_q)
    for i in range(5):
        mask = domains == i
        if np.any(mask):
            ax3.scatter(q_tsne[mask, 0], q_tsne[mask, 1],
                       c=colors[i], marker=markers[i],
                       label=f'Domain {i}', alpha=0.6)
    ax3.set_title('Query Features t-SNE')
    ax3.legend()
    
    # 4. 组合特征
    combined_features = np.concatenate([features_k, features_v, features_q], axis=1)
    combined_tsne = tsne.fit_transform(combined_features)
    for i in range(5):
        mask = domains == i
        if np.any(mask):
            ax4.scatter(combined_tsne[mask, 0], combined_tsne[mask, 1],
                       c=colors[i], marker=markers[i],
                       label=f'Domain {i}', alpha=0.6)
    ax4.set_title('Combined QKV Features t-SNE')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()