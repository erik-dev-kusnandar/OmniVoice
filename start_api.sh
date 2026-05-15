#!/bin/bash

# Navigasi ke folder OmniVoice
cd /home/ubuntu/OmniVoice

# Aktifkan virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: venv tidak ditemukan! Pastikan sudah install venv di folder ini."
    exit 1
fi

# Jalankan API di background menggunakan nohup
echo "------------------------------------------"
echo "Starting OmniVoice API at $(date)"
echo "------------------------------------------"
nohup python3 omnivoice_api.py > omnivoice_api.log 2>&1 &

# Simpan Process ID (PID)
echo $! > omnivoice_api.pid

echo "✅ OmniVoice API sedang berjalan di background (PID: $(cat omnivoice_api.pid))"
echo "📊 Untuk melihat log: tail -f omnivoice_api.log"
echo "🛑 Untuk mematikan: kill \$(cat omnivoice_api.pid)"
