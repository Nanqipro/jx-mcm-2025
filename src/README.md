# 代码索引

四个根级脚本是推荐入口：

| 脚本 | 对应任务 | 说明 |
| --- | --- | --- |
| `question1_video_features.py` | 问题一 | 视频抽帧、清晰度特征和综合模糊指数 |
| `question1_blur_regression.py` | 问题一 | 二次多项式 + 岭回归拟合 `blur_index` |
| `question2_continuous_visibility.py` | 问题二 | 连续模型、状态空间模型和可视化比较 |
| `question3_visibility_analysis.py` | 问题三 | 图像、气象与能见度的探索性关系分析 |

`experiments/` 保存比赛过程中形成的探索版和集成版脚本，便于追溯思路，但它们不是稳定 API，也未全部改造成命令行程序。新贡献优先修改上表中的推荐入口。
