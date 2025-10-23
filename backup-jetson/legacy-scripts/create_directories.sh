#!/bin/bash

# Script tạo thư mục cần thiết cho Jetson
echo "=== Tạo thư mục cho Jetson ==="

# Tạo các thư mục cần thiết
mkdir -p /home/edabk/Titanet/integration
mkdir -p /home/edabk/Titanet/integration/temp
mkdir -p /home/edabk/Titanet/integration/logs
mkdir -p /home/edabk/Titanet/integration/data
mkdir -p /home/edabk/Titanet/integration/dataset/test
mkdir -p /home/edabk/Titanet/integration/dataset/train

echo "✅ Đã tạo các thư mục:"
echo "  - /home/edabk/Titanet/integration (thư mục chính)"
echo "  - /home/edabk/Titanet/integration/temp (thư mục tạm)"
echo "  - /home/edabk/Titanet/integration/logs (log files)"
echo "  - /home/edabk/Titanet/integration/data (dữ liệu)"
echo "  - /home/edabk/Titanet/integration/dataset (dataset)"

# Set quyền
chmod -R 755 /home/edabk/Titanet/integration

echo ""
echo "🎉 Setup thư mục hoàn tất!"
echo ""
echo "Tiếp theo:"
echo "1. Copy tất cả files vào /home/edabk/Titanet/integration/"
echo "2. Chạy: python3 jetson_config.py để test"