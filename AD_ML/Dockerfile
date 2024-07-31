FROM ubuntu:latest
LABEL authors="98173"

# 使用官方的Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将requirements.txt复制到容器中
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录的内容复制到容器中
COPY . .

# 设置环境变量，Flask应用运行在调试模式
ENV FLASK_APP=api.py
ENV FLASK_ENV=development

# 暴露容器的端口
EXPOSE 5000

# 运行Flask应用
CMD ["flask", "run", "--host=0.0.0.0"]
