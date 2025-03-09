FastGPT 构建知识库
- 用高级编排功能添加流程处理非结构化数据，提取关键文本信息。
- 云端SaaS；Docker部署本地化部署（适合内部文件知识库）。Sealos：以k8s为核心的云操作系统，一键部署应用。
- 部署私有大模型。买算力使用AutoDL（用jupyterlab打开终端看显卡状态是英伟达smi配置），安装git-lfs后用modelscope下载ChatGLM-6B并克隆github，安装依赖，改model_path, port, 后启动模型。
- 把本地部署好的大模型接入FastGPT。

AutoGen
- 可以商业模型也可以开源模型
- AutoGen Studio组装配置：创建workflow（执行后所有agent的交互都会保存在json文件里，这些json就能驱动整个工作流程），playground测试后集成到前端网站应用（e.g. 视频自动生成文案工具）
- 多智能体User shell with human-in-the-loop: user proxy agent负责执行任务与反馈, assistant agent负责总体规划任务拆解，可以支持更多agent团队协同（生成图像，提取视频字幕，写文章等）
- LMStudio本地运行大模型，比如部署Qwen

在ChatGLM里微调LlaMa-Factory
- 创建ChatGLM.json文件输入数据集（e.g.你好我是客服）并添加在配置文件.json中，然后用CUDA Python启动UI打开LlaMa Board，填写微调细节配置参数，然后自动生成微调命令，开始微调。
- 训练结束后添加检查点路径（如果不添加，模型自认为是ChatGLM而不是客服），用HuggingFace加载模型测试效果。
- 训练好后export导出模型（检查点路径是生成训练后的检查点），进入模型路径中查看生成了很多新模型文件

DB-GPT
- 聚焦数据应用开发
- 生成一个Docker部署的MySQL服务器yaml文件，改环境变量后，docker启动服务器。用MySQL Workbench检验是否连接服务器。初始化数据后，去DB-GPT填上数据库信息。

Acknowledgements

[AI Agent智能体实战-极客时间-周文洋](https://time.geekbang.org/course/intro/100775901?tab=catalog)

<img src="https://github.com/user-attachments/assets/dce0937a-f364-46b6-b885-890c3f135d7f" width="50%" height="50%">

