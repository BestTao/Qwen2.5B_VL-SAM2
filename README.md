# Qwen2.5B_VL-SAM2
文本引导的视频分割


# 安装环境
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

pip install qwen-vl-utils[decord]==0.0.8 decord modelscope transformers==4.49.0 accelerate>=0.26.0

# 下载SAM2预训练模型和Qwen2.5B_VL-7B或32B
