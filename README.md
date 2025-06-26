# LLM-Study-Notes

# LLM系统学习路线：从 Hugging Face 到 Stanford CS324

## 一、基础实战阶段（Hugging Face LLM Course）

### 目标
- 搞懂LLM的基本原理和主流开源工具
- 能用Hugging Face库完成模型推理、微调和部署
- 熟悉Prompt Engineering和RAG等实战技巧

### 推荐课程与步骤
1. **Hugging Face LLM Course**  
   - [LLM Course 官方入口](https://huggingface.co/learn/llm-course/chapter1/1)
   - [LLM Course HuggingFace Transformers 手把手带你实战](https://www.bilibili.com/video/BV1Tm4y1J7EF/)
   - 重点学习：
     - LLM基础原理
     - Transformers库使用
     - Prompt Engineering
     - NLP任务实战
     - 高效微调与部署（Fine-tuning & Inference)
     - 低精度训练
     - 分布式训练
     - 常见实际应用（如问答、摘要、RAG等）
   - 每章都动手跑一遍官方notebook，建议整理代码笔记
2. **补充实战：LangChain入门**  
   - 了解如何用LangChain构建LLM应用
   - [LangChain官方快速入门](https://python.langchain.com/docs/get_started/introduction)

---

## 二、进阶原理&前沿应用阶段（过渡）

### 目标
- 理解Transformer、Self-Attention等底层机制
- 能读懂主流LLM论文（如GPT、BERT、LLaMA等）
- 掌握Prompt、对齐、安全等LLM应用难点

### 推荐学习材料
1. **CS224n（深度NLP）部分章节 CS106B C++与 数据结构与算法 CS161 算法设计 CS107 计算机组成原理**
   - 推荐：词嵌入、RNN/LSTM、Attention、Transformer、预训练语言模型（Lecture 7-11）
   - 推荐：深度学习Hyperparameter tuning, Regularization and Optimization、CNN、Reinforcement Learning, etc.
   - 推荐：C++算法设计与数据结构，对应Leetcode上面题型练习，结合CS106B和CS161并行学习
   - 推荐：CS107,系统学习计算机底层原理
   - [CS106B课程主页](https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1224/)
   - [CS106B课程视频](https://www.youtube.com/watch?v=Ua-31ucGAZ0&list=PLoCMsyE1cvdWiqgyzwAz_uGLSHsuYZlMX)
   - [CS106B课程github链接](https://github.com/zelenski/stanford-cpp-library)
   - [CS230课程github链接](https://github.com/maxim5/cs230-2018-autumn)
   - [CS230课程项目框架链接](https://cs230.stanford.edu/blog/)
   - [CS224n课程主页](https://web.stanford.edu/class/cs224n/)
   - [CS161课程主页](https://stanford-cs161.github.io/winter2025/)
   - [CS161课程github链接](https://github.com/adhaamehab/stanford-cs161)
   - [CS161课程存档](https://web.stanford.edu/class/archive/cs/cs161/cs161.1176/)
   - [CS107课程主页](https://web.stanford.edu/class/archive/cs/cs107/cs107.1224/)
   - [CS107课程视频🔗](https://www.youtube.com/watch?v=xuRkyNqqecc&list=PLoCMsyE1cvdWivlV-39KKsBKUX-4DvraN)
   - [CS107课程GitHub🔗](https://github.com/cs107e/cs107e.github.io.git)

2. **精选论文精读（建议）**
   - Attention is All You Need
   - BERT、GPT-2、LLaMA等论文
   - 可用arxiv-sanity、Paper Digest等工具快速阅读

3. **Hugging Face/OpenAI/LLM相关博客与工程文档**
   - [Hugging Face Blog](https://huggingface.co/blog/)
   - [OpenAI Blog](https://openai.com/blog/)

---

## 三、理论研究与创新应用阶段（Stanford CS324）

### 目标
- 系统掌握LLM理论、架构、训练、推理、对齐、安全、社会影响等
- 能独立完成LLM创新性项目或论文复现
- 跟进学术前沿，培养科研视角

### 学习流程
1. **跟随CS324 Calendar系统学习**
   - [CS324 Calendar](https://stanford-cs324.github.io/winter2022/calendar/)
   - 每周：
     - 预习Slides
     - 观看Lecture录像
     - 阅读推荐论文
     - 自主完成作业（Assignments）
   - 项目（Project）：可独立或组队完成一个LLM创新项目，参考 [Projects 页面](https://stanford-cs324.github.io/winter2022/projects/)

2. **参与社区讨论&复现优秀项目**
   - 在GitHub、Reddit等社区找同好讨论
   - 挑选历年CS324项目复现或创新，发布到自己的GitHub

3. **持续关注LLM相关顶会/新论文**
   - NeurIPS, ICML, ACL, ICLR等会议论文
   - arXiv每日订阅前沿LLM方向

---

## 路线总览

1. Hugging Face LLM Course（实战入门）
2. LangChain（应用开发）
3. CS224n部分精讲（理论打底）
4. 论文阅读（补齐理论/工程视野）
5. CS324全流程（系统前沿+创新项目）

---

## 小贴士
- 每阶段建议整理学习笔记与代码Demo
- 能力允许可尝试自己写blog总结或发布项目
- CS324部分作业/项目可与他人组队、交流提升效率
- 工程与理论相结合，持续关注社区与顶会动态

---

