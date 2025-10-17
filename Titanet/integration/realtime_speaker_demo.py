#!/usr/bin/env python3
"""
Real-time Speaker Recognition Demo
Thu âm liên tục từ microphone, phân đoạn VAD, và nhận dạng người nói
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from speaker_verification_pipeline import create_pipeline, RealTimeSpeakerRecognition

def main():
    print("🎤 Real-Time Speaker Recognition Demo")
    print("=====================================")
    
    # Check for enrolled speakers
    pipeline = create_pipeline()
    enrolled_speakers = list(pipeline.enrollment_db.keys())
    
    if not enrolled_speakers:
        print("⚠️  Không có người nào được đăng ký!")
        print("📝 Hãy đăng ký giọng nói trước:")
        print("   python3 voice_embedding_tool.py interactive")
        print("   > enroll")
        print("")
        response = input("Bạn có muốn tiếp tục không? (y/N): ")
        if response.lower() != 'y':
            return
    else:
        print(f"👥 Đã đăng ký {len(enrolled_speakers)} người:")
        for i, speaker in enumerate(enrolled_speakers, 1):
            print(f"   {i}. {speaker}")
        print("")
    
    # Get recording parameters
    print("⚙️  Cấu hình:")
    
    try:
        duration_input = input("Thời gian ghi (phút, Enter = không giới hạn): ").strip()
        duration = float(duration_input) if duration_input else None
    except ValueError:
        duration = None
    
    try:
        chunk_input = input("Độ dài đoạn phân tích (giây, mặc định 2.0): ").strip()
        chunk_duration = float(chunk_input) if chunk_input else 2.0
    except ValueError:
        chunk_duration = 2.0
    
    print("")
    print("🔧 Thiết lập:")
    print(f"   ⏱️  Thời gian: {'Không giới hạn' if duration is None else f'{duration} phút'}")
    print(f"   📊 Độ dài đoạn: {chunk_duration} giây")
    print(f"   🎯 Ngưỡng nhận dạng: {pipeline.config.similarity_threshold}")
    print(f"   🔊 VAD threshold: {pipeline.config.vad_threshold}")
    
    print("")
    input("Nhấn Enter để bắt đầu...")
    
    # Start recognition
    recognizer = RealTimeSpeakerRecognition(
        pipeline=pipeline, 
        chunk_duration=chunk_duration
    )
    
    try:
        recognizer.start_continuous_recognition(duration_minutes=duration)
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")

if __name__ == "__main__":
    main()