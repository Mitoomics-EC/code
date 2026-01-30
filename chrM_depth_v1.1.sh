#!/bin/bash

# 检查是否提供了样本目录作为命令行参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <sample_directory>"
    exit 1
fi

sample_dir=$1
insert_125="fragment_study/mis_10_splitedBy125/125_splited_sam_file"
depth_dir="depth_file"

# 进入目标目录
cd "${sample_dir}/${insert_125}"

# 创建深度文件目录
mkdir -p ../${depth_dir}

# 使用find命令查找文件大小超过1000的.sam文件
fils=$(find . -type f -name '*.sam' -size +1000c)

# 遍历所有符合条件的文件
for s in ${fils}; do
    # 计算深度信息并输出到对应的文本文件
    if samtools depth -d 500000000 -a -m 0 "${s}" > "../${depth_dir}/$(basename ${s}).txt"; then
        echo "Depth information for $(basename ${s}) has been successfully generated."
    else
        echo "Failed to generate depth information for $(basename ${s})."
    fi
done
