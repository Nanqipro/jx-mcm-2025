# 数据目录

本仓库不公开竞赛原始 AMOS 数据、机场视频或由题目方提供的第三方资料。这样处理是为了避免把“参加竞赛时可使用”误解为“可在开源仓库中再发布”。

## 本地目录约定

将自有或已获授权的数据放到下列位置；`data/private/` 已被 Git 忽略：

```text
data/
├── private/
│   ├── raw/amos_20200313/
│   │   ├── VIS_R06_12.his
│   │   ├── PTU_R06_12.his
│   │   └── WIND_R06_12.his
│   ├── processed/
│   │   ├── blur.csv
│   │   └── complete_synced_data.csv
│   └── video/                  # 本地视频或指向视频目录的链接
└── demo/                       # 脚本生成的合成演示数据
```

主流程使用的字段：

| 文件 | 主要字段 | 用途 |
| --- | --- | --- |
| `blur.csv` | `CREATEDATE`, `MOR_1A`, `TEMP`, `RH`, `DEWPOINT`, `WS2A`, `blur_index` | 问题一模糊指数回归、问题二连续模型 |
| `complete_synced_data.csv` | 图像清晰度特征、能见度、温湿压、风速风向、同步时间差 | 问题三多源关系分析 |

## 合成演示数据

无需竞赛数据即可生成结构兼容的演示文件：

```bash
python scripts/generate_demo_data.py
```

这些数据完全由确定性公式和伪随机噪声生成，只用于检查代码流程，不能复现论文数值，也不能作为模型有效性的证据。
