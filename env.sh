# env.sh

if [[ $(basename "$0") = "env.sh" ]]; then
    echo "Please source this script: 'source env.sh <env_name_or_path>'"
    return 1
fi

# 获取传入的环境路径或名称
ENV_PATH="$1"

if [[ -z "$ENV_PATH" ]]; then
    echo "Usage: source env.sh <env_path_or_name>"
    return 1
fi

# 绝对路径（确保 conda activate 正常）
ENV_PATH=$(realpath "$ENV_PATH")

# 如果虚拟环境还未创建
if [[ ! -f "$ENV_PATH/bin/pip" ]]; then
    conda create --prefix "$ENV_PATH" python=3.10 -c conda-forge || return 10
    conda activate "$ENV_PATH"

    # 需要使用conda安装的包
    # conda install -c conda-forge \

    pip -v install -r requirements.txt
fi

# 激活虚拟环境
conda activate "$ENV_PATH"

# 设置环境变量
export LD_LIBRARY_PATH="$ENV_PATH/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH=.

# 定义 update 函数
function update {
    pip -v install -r requirements.txt
}

# 执行传入的命令（如果有）
shift  # 去除第一个参数 ENV_PATH
"$@"
