# 图片制作需求 - 简明版

## 当前结论

**对于2页Poster版本：建议不添加图片** ❌

原因：
- 页数限制严格（≤2页）
- 表格已清晰展示数据
- 图片占用空间较大

---

## 如果需要制作，请制作以下图片：

### 1. 系统架构图 ⭐⭐⭐（最重要）

**文件名**：`system_architecture.pdf`

**尺寸**：宽度约 7cm（单栏）

**内容要点**：

```
离线阶段：
  [激活掩码M] → 计算共激活矩阵 → 谱聚类 → HCNS/LCNS分类

Level 1 (跨GPU划分):
  GPU 0: HCNS部分 + LCNS部分
  GPU 1: HCNS部分 + LCNS部分
  ...
  GPU n: HCNS部分 + LCNS部分

Level 2 (GPU内分配):
  每个GPU内部:
    TensorCore ← HCNS神经元
    CUDACore   ← LCNS神经元

Level 3 (流水线调度):
  Chunk k:   [计算] → [稀疏AllReduce]
  Chunk k+1:          [计算] → [稀疏AllReduce]
```

**制作建议**：
- 用方框表示各组件
- 用箭头表示数据流
- 颜色区分不同层级
- 保持简洁，避免过多细节

---

## 图片格式要求

- **格式**：PDF（矢量图，推荐）或 PNG（300 dpi以上）
- **宽度**：单栏约 7cm，双栏约 15cm
- **颜色**：黑白打印友好（不要只用颜色区分）
- **字体**：最小 8pt，清晰可读
- **线条**：不要太细，至少 0.5pt

---

## 工具推荐

### 在线工具（免费）：
- **draw.io**: https://app.diagrams.net/ （推荐）
- **Canva**: https://www.canva.com/

### 桌面工具：
- **Microsoft PowerPoint**
- **Apple Keynote**
- **Adobe Illustrator**

### 编程工具：
- **Python + matplotlib**（数据图）
- **LaTeX TikZ**（与论文集成）

---

## 制作流程（使用draw.io）

1. 访问 https://app.diagrams.net/
2. 选择"Blank Diagram"或"Basic Flowchart"
3. 按照上面的内容要点绘制
4. 导出为PDF: File → Export as → PDF
5. 保存到 `/doc/fig/system_architecture.pdf`

---

## 注意事项

1. **简洁**：不要堆砌细节，突出核心流程
2. **清晰**：标签和箭头要清楚
3. **一致**：使用统一的配色和风格
4. **专业**：参考顶级会议论文的图表风格

---

## 图片在论文中的使用

如果制作了图片，在论文中添加：

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{fig/system_architecture.pdf}
\caption{Three-level solver architecture}
\label{fig:architecture}
\end{figure}
```

---

## 总结

**当前建议**：不添加图片，使用表格即可

**如果需要**：优先制作系统架构图

**文件位置**：`/doc/fig/system_architecture.pdf`
