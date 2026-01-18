import os
import platform

# Pillow 호환성 패치 (ANTIALIAS -> LANCZOS)
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# ImageMagick 설정 (플랫폼별 자동 감지)
if platform.system() == 'Windows':
    # Windows: 일반적인 설치 경로들
    imagemagick_paths = [
        r'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe',
        r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe',
        r'C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe',
        r'C:\Program Files\ImageMagick\magick.exe',
        r'C:\Program Files (x86)\ImageMagick\magick.exe',
    ]
    for path in imagemagick_paths:
        if os.path.exists(path):
            os.environ['IMAGEMAGICK_BINARY'] = path
            break
else:
    # Linux/Mac
    os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'

import gradio as gr
import sys
import datetime
import numpy as np
import re
import tempfile

# 상위 폴더의 py 모듈 사용을 위해 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from helper import load_text_to_speech, load_voice_style, chunk_text  # type: ignore
import soundfile as sf
from docx import Document

# 전역 변수
tts_model = None
whisper_model = None
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
FONTS_DIR = os.path.join(ASSETS_DIR, 'fonts')

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)


def ensure_korean_font():
    """한글 폰트가 없으면 설치 - TTF 우선"""
    font_path = os.path.join(FONTS_DIR, 'NotoSansKR-Bold.ttf')

    # 이미 폰트가 있으면 스킵
    if os.path.exists(font_path) and os.path.getsize(font_path) > 100000:
        print(f"한글 폰트 확인됨: {font_path}")
        return font_path

    # TTF 시스템 폰트 우선 확인 (PIL에서 직접 로드 가능)
    ttf_fonts = [
        '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
        '/usr/share/fonts/nanum/NanumGothicBold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansKR-Bold.otf',
        'C:/Windows/Fonts/NotoSansKR-Bold.ttf',
        'C:/Windows/Fonts/malgunbd.ttf',
    ]

    for sys_font in ttf_fonts:
        if os.path.exists(sys_font):
            print(f"시스템 TTF 폰트 발견: {sys_font}")
            return sys_font

    # TTC 폰트 (PIL에서 index 필요)
    ttc_fonts = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc',
    ]

    for sys_font in ttc_fonts:
        if os.path.exists(sys_font):
            print(f"시스템 TTC 폰트 발견: {sys_font}")
            return sys_font

    # Linux에서 apt로 설치 시도
    print("한글 폰트 설치 시도 중...")
    try:
        import subprocess
        # fonts-nanum 설치 (TTF 파일 제공)
        subprocess.run(
            ['apt-get', 'install', '-y', 'fonts-nanum'],
            capture_output=True, text=True, timeout=120
        )
        # fc-cache 실행
        subprocess.run(['fc-cache', '-f', '-v'], capture_output=True, timeout=60)

        # 설치 후 TTF 다시 확인
        for sys_font in ttf_fonts:
            if os.path.exists(sys_font):
                print(f"apt 설치 후 TTF 폰트 발견: {sys_font}")
                return sys_font

        # TTC 확인
        for sys_font in ttc_fonts:
            if os.path.exists(sys_font):
                print(f"apt 설치 후 TTC 폰트 발견: {sys_font}")
                return sys_font
    except Exception as e:
        print(f"apt 설치 실패: {e}")

    # pip로 폰트 패키지 설치 시도
    try:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'fonts', 'font-noto-sans-kr'],
                      capture_output=True, timeout=60)
    except Exception as e:
        print(f"pip 폰트 설치 실패: {e}")

    # 다운로드 시도 (여러 URL)
    font_urls = [
        "https://raw.githubusercontent.com/nickmass/font-patcher/main/fonts/NotoSansKR-Bold.ttf",
        "https://cdn.jsdelivr.net/gh/nickmass/font-patcher/fonts/NotoSansKR-Bold.ttf",
    ]

    for font_url in font_urls:
        try:
            import urllib.request
            print(f"폰트 다운로드 시도: {font_url}")
            urllib.request.urlretrieve(font_url, font_path)
            if os.path.exists(font_path) and os.path.getsize(font_path) > 100000:
                print(f"한글 폰트 다운로드 완료: {font_path}")
                return font_path
        except Exception as e:
            print(f"다운로드 실패: {e}")
            continue

    print("경고: 한글 폰트를 찾을 수 없습니다!")
    return None


# 앱 시작 시 폰트 확인
KOREAN_FONT_PATH = ensure_korean_font()


def check_gpu_available():
    """GPU 사용 가능 여부 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU 감지됨: {gpu_name}")
            return True
    except ImportError:
        pass
    print("GPU 없음 - CPU 모드로 실행")
    return False


def init_tts(use_gpu=False):
    """TTS 모델 초기화 (CPU 모드 기본)"""
    global tts_model
    if tts_model is None:
        onnx_dir = os.path.join(ASSETS_DIR, 'onnx')
        tts_model = load_text_to_speech(onnx_dir, use_gpu=use_gpu)
        mode = "GPU" if use_gpu else "CPU"
        print(f"TTS 모델 로드 완료! ({mode})")
    return tts_model


def init_whisper():
    """Whisper 모델 초기화 (지연 로딩)"""
    global whisper_model
    if whisper_model is None:
        import whisper  # type: ignore
        print("Whisper 모델 로드 중...")
        whisper_model = whisper.load_model("base")
        print("Whisper 모델 로드 완료!")
    return whisper_model


def analyze_audio_with_whisper(audio_path, language='ko'):
    """Whisper로 오디오 분석하여 단어/구간별 타임스탬프 추출"""
    model = init_whisper()

    lang_map = {
        'ko': 'ko', 'en': 'en', 'es': 'es', 'pt': 'pt', 'fr': 'fr'
    }
    whisper_lang = lang_map.get(language, 'ko')

    result = model.transcribe(
        audio_path,
        language=whisper_lang,
        word_timestamps=True,
        verbose=False
    )
    return result


def match_subtitles_to_audio(whisper_result, subtitle_lines, audio_duration):
    """Whisper 분석 결과와 자막 텍스트를 매칭하여 타임코드 생성

    Whisper 세그먼트의 시작/끝 시간을 기준으로 자막 라인을 균등 배분
    """
    subtitle_timings = []
    segments = whisper_result.get('segments', [])

    if not subtitle_lines:
        return subtitle_timings

    total_lines = len(subtitle_lines)

    # Whisper 세그먼트가 있으면 세그먼트 시간 기반으로 배분
    if segments:
        # 전체 음성 구간 (첫 세그먼트 시작 ~ 마지막 세그먼트 끝)
        speech_start = segments[0]['start']
        speech_end = segments[-1]['end']
        speech_duration = speech_end - speech_start

        # 각 자막 라인의 길이(글자수) 기반으로 시간 배분
        line_lengths = [len(line) for line in subtitle_lines]
        total_chars = sum(line_lengths)

        current_time = speech_start
        for i, line in enumerate(subtitle_lines):
            # 글자 수 비율로 시간 배분
            char_ratio = line_lengths[i] / total_chars if total_chars > 0 else 1 / total_lines
            line_duration = speech_duration * char_ratio

            # 최소 0.5초, 최대는 제한 없음
            line_duration = max(0.5, line_duration)

            start_time = current_time
            end_time = min(current_time + line_duration, audio_duration)

            subtitle_timings.append({
                'text': line,
                'start': start_time,
                'end': end_time
            })

            current_time = end_time

            print(f"자막 타이밍: [{start_time:.2f}s - {end_time:.2f}s] {line[:30]}")
    else:
        # Whisper 세그먼트가 없으면 균등 분배
        time_per_line = audio_duration / total_lines
        for i, line in enumerate(subtitle_lines):
            subtitle_timings.append({
                'text': line,
                'start': i * time_per_line,
                'end': (i + 1) * time_per_line
            })

    # 마지막 자막은 오디오 끝까지
    if subtitle_timings:
        subtitle_timings[-1]['end'] = audio_duration

    return subtitle_timings


def get_max_length(lang):
    """언어별 최대 청크 길이 반환 (Supertonic 정책)"""
    return 120 if lang == "ko" else 300


def get_voice_list():
    """사용 가능한 음성 목록 반환"""
    voice_dir = os.path.join(ASSETS_DIR, 'voice_styles')
    voices = []

    if os.path.exists(voice_dir):
        for f in sorted(os.listdir(voice_dir)):
            if f.endswith('.json'):
                name = f.replace('.json', '')
                label = f"여성 {name[1]}" if name.startswith('F') else f"남성 {name[1]}"
                voices.append(f"{label} ({name})")

    return voices


def get_voice_file(voice_label):
    """음성 라벨에서 파일명 추출"""
    # "여성 1 (F1)" -> "F1.json"
    match = re.search(r'\(([^)]+)\)', voice_label)
    if match:
        return f"{match.group(1)}.json"
    return "F1.json"


def read_text_file(file_path):
    """TXT 또는 DOCX 파일 읽기"""
    if file_path is None:
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            return f"지원하지 않는 파일 형식입니다: {ext}"
    except Exception as e:
        return f"파일 읽기 오류: {str(e)}"


def synthesize_speech(text, voice_label, language, speed, total_step, progress=gr.Progress(), output_name=None):
    """음성 합성"""
    if not text or not text.strip():
        return None, "텍스트를 입력해주세요."

    try:
        progress(0.05, desc="텍스트 분석 중...")

        voice_file = get_voice_file(voice_label)
        max_len = get_max_length(language)
        chunks = chunk_text(text, max_len=max_len)
        total_chunks = len(chunks) if chunks else 1

        progress(0.15, desc="TTS 모델 로드 중...")
        tts = init_tts()

        voice_path = os.path.join(ASSETS_DIR, 'voice_styles', voice_file)
        style = load_voice_style([voice_path], verbose=False)

        all_audio = []
        total_duration = 0.0

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            prog = 0.2 + (i / total_chunks) * 0.6
            preview = chunk[:30] + '...' if len(chunk) > 30 else chunk
            progress(prog, desc=f'[{i + 1}/{total_chunks}] {preview}')

            wav, duration = tts(chunk, language, style, int(total_step), float(speed))
            w = wav[0, :int(tts.sample_rate * duration[0].item())]
            all_audio.append(w)
            total_duration += duration[0].item()

            if i < total_chunks - 1:
                silence = np.zeros(int(0.3 * tts.sample_rate), dtype=np.float32)
                all_audio.append(silence)
                total_duration += 0.3

        progress(0.85, desc="오디오 병합 중...")

        if len(all_audio) > 1:
            combined = np.concatenate(all_audio)
        else:
            combined = all_audio[0] if all_audio else np.array([], dtype=np.float32)

        progress(0.90, desc="파일 저장 중...")
        # 출력 파일명: 대본 파일명 또는 타임스탬프
        if output_name:
            filename = f"{output_name}.wav"
        else:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tts_{timestamp}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        sf.write(filepath, combined, tts.sample_rate)

        progress(1.0, desc="완료!")

        return filepath, f"✅ 음성 생성 완료!\n파일: {filename}\n길이: {total_duration:.1f}초"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류 발생: {str(e)}"


def create_video(tts_text, subtitle_text, voice_label, language, speed, total_step,
                 background_file, resolution, font_size, subtitle_position,
                 subtitle_offset_x, subtitle_offset_y,
                 use_subtitle_bg, subtitle_bg_opacity, subtitle_bg_padding,
                 progress=gr.Progress(), output_name=None):
    """영상 생성"""
    print(f"=== create_video 호출 ===")
    print(f"use_subtitle_bg={use_subtitle_bg} (type={type(use_subtitle_bg)})")
    print(f"subtitle_position={subtitle_position}, offset_x={subtitle_offset_x}%, offset_y={subtitle_offset_y}%")
    print(f"font_size={font_size}, opacity={subtitle_bg_opacity}, padding={subtitle_bg_padding}")

    if not tts_text or not tts_text.strip():
        return None, "TTS 텍스트를 입력해주세요."

    if not subtitle_text or not subtitle_text.strip():
        subtitle_text = tts_text

    # 기본값 처리 (None이거나 범위 밖이면 기본값 사용)
    try:
        font_size = int(font_size) if font_size is not None else 70
        if font_size < 10 or font_size > 200:
            font_size = 70
    except (ValueError, TypeError):
        font_size = 70

    try:
        subtitle_bg_opacity = float(subtitle_bg_opacity) if subtitle_bg_opacity is not None else 0.6
        if subtitle_bg_opacity < 0.1 or subtitle_bg_opacity > 1.0:
            subtitle_bg_opacity = 0.6
    except (ValueError, TypeError):
        subtitle_bg_opacity = 0.6

    try:
        subtitle_bg_padding = int(subtitle_bg_padding) if subtitle_bg_padding is not None else 20
        if subtitle_bg_padding < 0 or subtitle_bg_padding > 100:
            subtitle_bg_padding = 20
    except (ValueError, TypeError):
        subtitle_bg_padding = 20

    # X/Y 오프셋 처리 (% 단위, -50 ~ 50 범위)
    try:
        subtitle_offset_x = float(subtitle_offset_x) if subtitle_offset_x is not None else 0
        if subtitle_offset_x < -50 or subtitle_offset_x > 50:
            subtitle_offset_x = 0
    except (ValueError, TypeError):
        subtitle_offset_x = 0

    try:
        subtitle_offset_y = float(subtitle_offset_y) if subtitle_offset_y is not None else 0
        if subtitle_offset_y < -50 or subtitle_offset_y > 50:
            subtitle_offset_y = 0
    except (ValueError, TypeError):
        subtitle_offset_y = 0

    resolution = resolution if resolution else "1920x1080"

    try:
        from moviepy.editor import (  # type: ignore
            ImageClip, VideoFileClip, AudioFileClip,
            CompositeVideoClip, ColorClip
        )

        progress(0.05, desc="준비 중...")

        video_width, video_height = map(int, resolution.split('x'))
        voice_file = get_voice_file(voice_label)

        # 배경 파일 처리
        background_path = None
        background_type = None
        if background_file is not None:
            background_path = background_file
            ext = os.path.splitext(background_file)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                background_type = 'video'
            else:
                background_type = 'image'

        # 음성 생성
        progress(0.10, desc="TTS 모델 로드 중...")
        tts = init_tts()

        voice_path = os.path.join(ASSETS_DIR, 'voice_styles', voice_file)
        style = load_voice_style([voice_path], verbose=False)

        max_len = get_max_length(language)
        chunks = chunk_text(tts_text, max_len=max_len)
        if not chunks:
            chunks = [tts_text]

        total_chunks = len(chunks)
        all_audio = []
        audio_duration = 0.0

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            prog = 0.15 + (i / total_chunks) * 0.25
            preview = chunk[:20] + '...' if len(chunk) > 20 else chunk
            progress(prog, desc=f'음성 [{i + 1}/{total_chunks}] {preview}')

            wav, duration = tts(chunk, language, style, int(total_step), float(speed))
            w = wav[0, :int(tts.sample_rate * duration[0].item())]
            all_audio.append(w)
            audio_duration += duration[0].item()

            if i < total_chunks - 1:
                silence = np.zeros(int(0.3 * tts.sample_rate), dtype=np.float32)
                all_audio.append(silence)
                audio_duration += 0.3

        if len(all_audio) > 1:
            combined_audio = np.concatenate(all_audio)
        else:
            combined_audio = all_audio[0] if all_audio else np.array([], dtype=np.float32)

        progress(0.40, desc="오디오 파일 저장 중...")
        temp_audio_path = os.path.join(TEMP_DIR, "temp_audio.wav")
        sf.write(temp_audio_path, combined_audio, tts.sample_rate)

        # Whisper 분석
        progress(0.42, desc="Whisper 모델 로드 중...")
        subtitle_lines = [line.strip() for line in subtitle_text.split('\n') if line.strip()]

        if subtitle_lines:
            progress(0.45, desc="음성 분석 중... (Whisper)")
            try:
                whisper_result = analyze_audio_with_whisper(temp_audio_path, language)
                progress(0.50, desc="자막 타임코드 생성 중...")
                subtitle_timings = match_subtitles_to_audio(
                    whisper_result, subtitle_lines, audio_duration
                )
            except Exception as e:
                print(f"Whisper 분석 실패, 균등 분배 사용: {e}")
                time_per_line = audio_duration / len(subtitle_lines)
                subtitle_timings = [
                    {'text': line, 'start': i * time_per_line, 'end': (i + 1) * time_per_line}
                    for i, line in enumerate(subtitle_lines)
                ]
        else:
            subtitle_timings = []

        # 배경 클립 생성
        progress(0.55, desc="배경 영상 준비 중...")

        if background_path and background_type == 'video':
            bg_clip = VideoFileClip(background_path)
            if bg_clip.duration < audio_duration:
                bg_clip = bg_clip.loop(duration=audio_duration)
            else:
                bg_clip = bg_clip.subclip(0, audio_duration)
            bg_clip = bg_clip.resize((video_width, video_height))
        elif background_path and background_type == 'image':
            bg_clip = ImageClip(background_path).set_duration(audio_duration)
            bg_clip = bg_clip.resize((video_width, video_height))
        else:
            bg_clip = ColorClip(size=(video_width, video_height), color=(26, 26, 46)).set_duration(audio_duration)

        # 자막 위치 계산
        def get_subtitle_pos(pos, width, height, fsize, offset_x_pct, offset_y_pct):
            """자막 위치 계산 (오프셋은 해상도의 % 단위)"""
            margin = 50
            # 오프셋 픽셀 계산 (해상도의 %)
            offset_x_px = int(width * offset_x_pct / 100)
            offset_y_px = int(height * offset_y_pct / 100)

            positions = {
                '상단-왼쪽': (margin, margin),
                '상단-중앙': ('center', margin),
                '상단-오른쪽': (width - margin, margin),
                '중앙-왼쪽': (margin, 'center'),
                '중앙': ('center', 'center'),
                '중앙-오른쪽': (width - margin, 'center'),
                '하단-왼쪽': (margin, height - margin - fsize),
                '하단-중앙': ('center', height - margin - fsize),
                '하단-오른쪽': (width - margin, height - margin - fsize),
            }
            base_pos = positions.get(pos, ('center', height - margin - fsize))

            # 오프셋 적용 (center인 경우 픽셀로 변환 후 오프셋 적용)
            final_x = base_pos[0]
            final_y = base_pos[1]

            if final_x == 'center':
                final_x = width // 2 + offset_x_px
            else:
                final_x = final_x + offset_x_px

            if final_y == 'center':
                final_y = height // 2 + offset_y_px
            else:
                final_y = final_y + offset_y_px

            return (final_x, final_y)

        txt_position = get_subtitle_pos(subtitle_position, video_width, video_height, font_size, subtitle_offset_x, subtitle_offset_y)

        # 자막 클립 생성 (PIL 기반 - ImageMagick 폰트 문제 우회)
        progress(0.60, desc="자막 클립 생성 중...")
        subtitle_clips = []

        # PIL 폰트 로드 (한 번만)
        from PIL import Image as PILImage, ImageDraw, ImageFont

        # 폰트 찾기 - TTF 우선
        font_candidates = [
            os.path.join(FONTS_DIR, 'NotoSansKR-Bold.ttf'),
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf',
            'C:/Windows/Fonts/NotoSansKR-Bold.ttf',
            'C:/Windows/Fonts/malgunbd.ttf',
        ]
        ttc_candidates = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc',
        ]

        pil_font = None
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    pil_font = ImageFont.truetype(font_path, font_size)
                    print(f"PIL 폰트 로드 성공 (TTF): {font_path}")
                    break
                except Exception as e:
                    print(f"TTF 폰트 로드 실패: {font_path} - {e}")

        if pil_font is None:
            for font_path in ttc_candidates:
                if os.path.exists(font_path):
                    try:
                        pil_font = ImageFont.truetype(font_path, font_size, index=1)
                        print(f"PIL 폰트 로드 성공 (TTC): {font_path}")
                        break
                    except Exception as e:
                        print(f"TTC 폰트 로드 실패: {font_path} - {e}")

        if pil_font is None:
            print("모든 폰트 로드 실패, 기본 폰트 사용")
            pil_font = ImageFont.load_default()

        for i, timing in enumerate(subtitle_timings):
            line = timing['text']
            start_time = timing['start']
            end_time = timing['end']

            if not line:
                continue

            if i % 5 == 0:
                prog = 0.60 + (i / len(subtitle_timings)) * 0.15
                progress(prog, desc=f'자막 클립 [{i + 1}/{len(subtitle_timings)}]')

            try:
                # PIL로 자막 이미지 생성
                # 텍스트 크기 측정
                dummy_img = PILImage.new('RGBA', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_img)
                bbox = dummy_draw.textbbox((0, 0), line, font=pil_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 외곽선 두께
                outline_width = 3
                padding = int(subtitle_bg_padding) if use_subtitle_bg else 10

                # 이미지 크기: 자막 배경 사용 시 화면 전체 너비
                if use_subtitle_bg:
                    img_width = video_width
                    img_height = text_height + padding * 2 + outline_width * 2
                else:
                    img_width = text_width + outline_width * 2 + 20
                    img_height = text_height + outline_width * 2 + 10

                # RGBA 이미지 생성 (투명 배경)
                subtitle_img = PILImage.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(subtitle_img)

                # 자막 배경 박스 (선택적)
                if use_subtitle_bg:
                    alpha = int(255 * subtitle_bg_opacity)
                    draw.rectangle([0, 0, img_width, img_height], fill=(0, 0, 0, alpha))

                # 텍스트 위치 계산 (이미지 내 중앙)
                text_x = (img_width - text_width) // 2
                text_y = (img_height - text_height) // 2

                # 외곽선 (검정)
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), line, font=pil_font, fill=(0, 0, 0, 255))

                # 본문 (흰색)
                draw.text((text_x, text_y), line, font=pil_font, fill=(255, 255, 255, 255))

                # PIL 이미지를 numpy 배열로 변환
                img_array = np.array(subtitle_img)

                # ImageClip 생성
                txt_clip = ImageClip(img_array, ismask=False, transparent=True)
                txt_clip = txt_clip.set_duration(end_time - start_time)

                # 자막 위치 계산
                if use_subtitle_bg:
                    # 배경 사용 시: 가로는 0 (전체 너비), 세로는 txt_position 기준
                    clip_x = 0
                    if txt_position[1] == 'center':
                        clip_y = (video_height - img_height) // 2
                    else:
                        clip_y = txt_position[1] - padding if isinstance(txt_position[1], int) else video_height - img_height - 50
                else:
                    # 배경 미사용 시: txt_position 기준
                    if txt_position[0] == 'center':
                        clip_x = (video_width - img_width) // 2
                    else:
                        clip_x = txt_position[0] if isinstance(txt_position[0], int) else 50

                    if txt_position[1] == 'center':
                        clip_y = (video_height - img_height) // 2
                    else:
                        clip_y = txt_position[1] if isinstance(txt_position[1], int) else video_height - img_height - 50

                txt_clip = txt_clip.set_position((clip_x, clip_y))
                txt_clip = txt_clip.set_start(start_time).set_end(end_time)
                subtitle_clips.append(txt_clip)
                print(f"자막 추가 (PIL): [{start_time:.2f}s - {end_time:.2f}s] {line[:20]}...")

            except Exception as e:
                import traceback
                print(f"자막 클립 생성 실패 [{i}]: {e}")
                traceback.print_exc()

        print(f"총 자막 클립 수: {len(subtitle_clips)}")
        progress(0.75, desc="영상 합성 중...")
        final_clip = CompositeVideoClip([bg_clip] + subtitle_clips)

        progress(0.78, desc="오디오 추가 중...")
        audio_clip = AudioFileClip(temp_audio_path)
        final_clip = final_clip.set_audio(audio_clip)

        progress(0.80, desc="영상 인코딩 중... (시간이 걸릴 수 있습니다)")
        # 출력 파일명: 대본 파일명 또는 타임스탬프
        if output_name:
            filename = f"{output_name}.mp4"
        else:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"video_{timestamp}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)

        final_clip.write_videofile(
            filepath,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )

        # 리소스 정리
        final_clip.close()
        audio_clip.close()
        if background_path and background_type == 'video':
            bg_clip.close()

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        progress(1.0, desc="완료!")

        return filepath, f"✅ 영상 생성 완료!\n파일: {filename}\n길이: {audio_duration:.1f}초"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류 발생: {str(e)}"


def load_tts_text(file):
    """TTS 텍스트 파일 로드"""
    if file is None:
        return ""
    return read_text_file(file.name)


def load_subtitle_text(file):
    """자막 텍스트 파일 로드"""
    if file is None:
        return ""
    return read_text_file(file.name)


def generate_preview(subtitle_text, background_file, resolution, font_size, subtitle_position,
                     subtitle_offset_x, subtitle_offset_y,
                     use_subtitle_bg, subtitle_bg_opacity, subtitle_bg_padding):
    """자막이 포함된 미리보기 이미지 생성"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # 기본값 처리 (None이거나 범위 밖이면 기본값 사용)
        try:
            font_size = int(font_size) if font_size is not None else 70
            if font_size < 10 or font_size > 200:
                font_size = 70
        except (ValueError, TypeError):
            font_size = 70

        try:
            subtitle_bg_opacity = float(subtitle_bg_opacity) if subtitle_bg_opacity is not None else 0.6
            if subtitle_bg_opacity < 0.1 or subtitle_bg_opacity > 1.0:
                subtitle_bg_opacity = 0.6
        except (ValueError, TypeError):
            subtitle_bg_opacity = 0.6

        try:
            subtitle_bg_padding = int(subtitle_bg_padding) if subtitle_bg_padding is not None else 20
            if subtitle_bg_padding < 0 or subtitle_bg_padding > 100:
                subtitle_bg_padding = 20
        except (ValueError, TypeError):
            subtitle_bg_padding = 20

        # X/Y 오프셋 처리 (% 단위)
        try:
            subtitle_offset_x = float(subtitle_offset_x) if subtitle_offset_x is not None else 0
            if subtitle_offset_x < -50 or subtitle_offset_x > 50:
                subtitle_offset_x = 0
        except (ValueError, TypeError):
            subtitle_offset_x = 0

        try:
            subtitle_offset_y = float(subtitle_offset_y) if subtitle_offset_y is not None else 0
            if subtitle_offset_y < -50 or subtitle_offset_y > 50:
                subtitle_offset_y = 0
        except (ValueError, TypeError):
            subtitle_offset_y = 0

        resolution = resolution if resolution else "1920x1080"

        video_width, video_height = map(int, resolution.split('x'))

        # 배경 이미지 생성
        if background_file is not None:
            ext = os.path.splitext(background_file.name)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                # 영상에서 첫 프레임 추출
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(background_file.name)
                frame = clip.get_frame(0)
                clip.close()
                bg_img = Image.fromarray(frame)
                bg_img = bg_img.resize((video_width, video_height), Image.LANCZOS)
            else:
                # 이미지 파일
                bg_img = Image.open(background_file.name)
                bg_img = bg_img.resize((video_width, video_height), Image.LANCZOS)
                bg_img = bg_img.convert('RGBA')
        else:
            # 기본 배경 (어두운 색)
            bg_img = Image.new('RGBA', (video_width, video_height), (26, 26, 46, 255))

        # 자막 텍스트 처리
        if not subtitle_text or not subtitle_text.strip():
            subtitle_text = "자막 미리보기 텍스트"

        # 첫 번째 줄만 미리보기에 표시
        first_line = subtitle_text.strip().split('\n')[0]

        # 폰트 찾기 - TTF 우선, TTC는 인덱스 필요
        font_candidates = [
            # TTF 파일 우선 (PIL에서 직접 로드 가능)
            os.path.join(FONTS_DIR, 'NotoSansKR-Bold.ttf'),
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf',
            'C:/Windows/Fonts/NotoSansKR-Bold.ttf',
            'C:/Windows/Fonts/malgunbd.ttf',
        ]

        # TTC 파일 (인덱스 필요)
        ttc_candidates = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc',
        ]

        font = None

        # TTF 먼저 시도
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"미리보기 폰트 로드 성공 (TTF): {font_path}")
                    break
                except Exception as e:
                    print(f"TTF 폰트 로드 실패: {font_path} - {e}")
                    continue

        # TTF 실패시 TTC 시도 (인덱스 0 = 한국어)
        if font is None:
            for font_path in ttc_candidates:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size, index=1)  # index 1 = Korean
                        print(f"미리보기 폰트 로드 성공 (TTC): {font_path}")
                        break
                    except Exception as e:
                        print(f"TTC 폰트 로드 실패: {font_path} - {e}")
                        continue

        # 모두 실패시 기본 폰트
        if font is None:
            print("모든 폰트 로드 실패, 기본 폰트 사용")
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(bg_img)

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), first_line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 자막 위치 계산
        margin = 50
        # 오프셋 픽셀 계산 (해상도의 %)
        offset_x_px = int(video_width * subtitle_offset_x / 100)
        offset_y_px = int(video_height * subtitle_offset_y / 100)

        positions = {
            '상단-왼쪽': (margin, margin),
            '상단-중앙': ((video_width - text_width) // 2, margin),
            '상단-오른쪽': (video_width - text_width - margin, margin),
            '중앙-왼쪽': (margin, (video_height - text_height) // 2),
            '중앙': ((video_width - text_width) // 2, (video_height - text_height) // 2),
            '중앙-오른쪽': (video_width - text_width - margin, (video_height - text_height) // 2),
            '하단-왼쪽': (margin, video_height - text_height - margin),
            '하단-중앙': ((video_width - text_width) // 2, video_height - text_height - margin),
            '하단-오른쪽': (video_width - text_width - margin, video_height - text_height - margin),
        }
        base_x, base_y = positions.get(subtitle_position, positions['하단-중앙'])
        # 오프셋 적용
        text_x = base_x + offset_x_px
        text_y = base_y + offset_y_px

        # 자막 배경 박스 그리기 (화면 전체 너비, 자막 높이 + 패딩)
        if use_subtitle_bg:
            padding = int(subtitle_bg_padding)
            # 가로: 화면 전체 + 여유, 세로: 자막 높이 + 패딩*2
            bg_h = text_height + padding * 2
            bg_x1 = -5
            bg_x2 = video_width + 5

            # 배경 박스의 Y 위치 계산 (자막이 박스 정중앙에 오도록)
            bg_y1 = text_y - padding
            bg_y2 = bg_y1 + bg_h

            # 반투명 배경
            overlay = Image.new('RGBA', bg_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            alpha = int(255 * subtitle_bg_opacity)
            overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, alpha))
            bg_img = Image.alpha_composite(bg_img.convert('RGBA'), overlay)

        # 텍스트 그리기 (검정 외곽선 + 흰색 본문)
        draw = ImageDraw.Draw(bg_img)

        # 외곽선 (검정, 두께 3)
        outline_width = 3
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), first_line, font=font, fill=(0, 0, 0, 255))

        # 본문 (흰색 고정)
        draw.text((text_x, text_y), first_line, font=font, fill=(255, 255, 255, 255))

        # 미리보기 저장
        preview_path = os.path.join(TEMP_DIR, "preview.png")
        bg_img.convert('RGB').save(preview_path)

        return preview_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# Gradio UI 구성 (심플 디자인)
def create_ui():
    voices = get_voice_list()
    if not voices:
        voices = ["음성 파일 없음"]

    languages = ["한국어", "English", "Español", "Português", "Français"]

    resolutions = ["1920x1080", "1280x720", "3840x2160", "1080x1920", "720x1280"]

    subtitle_positions = ["중앙", "하단-중앙", "상단-중앙"]

    with gr.Blocks(title="Supertonic TTS") as demo:
        gr.Markdown("# Supertonic TTS")

        # 1행: 파일 업로드 (대본, 자막, 배경)
        with gr.Row():
            tts_file = gr.File(
                label="대본 파일 (TXT/DOCX)",
                file_types=[".txt", ".docx"]
            )
            subtitle_file = gr.File(
                label="자막 파일 (TXT/DOCX)",
                file_types=[".txt", ".docx"]
            )
            background_file = gr.File(
                label="배경 (이미지/영상)",
                file_types=["image", "video"]
            )

        # 2행: 텍스트 입력 + 미리보기
        with gr.Row():
            tts_text = gr.Textbox(
                label="대본 (음성 변환용)",
                placeholder="음성으로 변환할 텍스트를 입력하거나 파일을 첨부하세요...",
                lines=10
            )
            subtitle_text = gr.Textbox(
                label="자막 (비워두면 대본 사용)",
                placeholder="화면에 표시될 자막...",
                lines=10
            )
            preview_image = gr.Image(label="미리보기", height=280)

        # 3행: 음성 설정 + 영상 설정 + 상태 + 생성버튼 (한 줄)
        with gr.Row():
            voice_select = gr.Dropdown(choices=voices, value=voices[0] if voices else None, label="음성", scale=2)
            speed_slider = gr.Number(value=1.0, label="속도", minimum=0.5, maximum=2.0, step=0.1, scale=1)
            lang_select = gr.Dropdown(choices=languages, value="한국어", label="언어", scale=1)
            step_slider = gr.Number(value=5, label="품질", minimum=1, maximum=10, step=1, scale=1)
            # 영상 설정 (배경 첨부 시만 사용됨)
            resolution_select = gr.Dropdown(choices=resolutions, value="1920x1080", label="해상도", visible=False, scale=2)
            font_size_slider = gr.Number(value=70, label="폰트", step=5, visible=False, scale=1)
            position_select = gr.Dropdown(choices=subtitle_positions, value="중앙", label="자막위치", visible=False, scale=2)
            subtitle_offset_x = gr.Number(value=0, label="X오프셋(%)", step=1, visible=False, scale=1)
            subtitle_offset_y = gr.Number(value=0, label="Y오프셋(%)", step=1, visible=False, scale=1)
            use_subtitle_bg = gr.Checkbox(label="자막배경 사용", value=True, visible=False, scale=2, min_width=120)
            subtitle_bg_opacity = gr.Number(value=0.6, label="투명도", step=0.1, visible=False, scale=1)
            subtitle_bg_padding = gr.Number(value=20, label="여백", step=5, visible=False, scale=1)
            status_output = gr.Textbox(label="상태", interactive=False, scale=2)
            generate_btn = gr.Button("생성하기", variant="primary", scale=1)

        # 4행: 결과
        with gr.Row():
            audio_output = gr.Audio(label="결과 음성", type="filepath")
            video_output = gr.Video(label="결과 영상", visible=False)

        # 다운로드 파일 (Kaggle용)
        with gr.Row():
            download_file = gr.File(label="📥 다운로드", visible=False)

        # 이벤트 연결
        def get_lang_code(lang_name):
            lang_map = {"한국어": "ko", "English": "en", "Español": "es", "Português": "pt", "Français": "fr"}
            return lang_map.get(lang_name, "ko")

        # 배경 파일 첨부 시 영상 설정 표시/숨김
        def toggle_video_settings(file):
            visible = file is not None
            return [gr.update(visible=visible)] * 9  # 8개 영상설정 + 1개 영상출력

        background_file.change(
            fn=toggle_video_settings,
            inputs=[background_file],
            outputs=[resolution_select, font_size_slider, position_select,
                     subtitle_offset_x, subtitle_offset_y,
                     use_subtitle_bg, subtitle_bg_opacity, subtitle_bg_padding, video_output]
        )

        # 미리보기 입력 컴포넌트 리스트
        preview_inputs = [
            subtitle_text, background_file, resolution_select,
            font_size_slider, position_select,
            subtitle_offset_x, subtitle_offset_y,
            use_subtitle_bg, subtitle_bg_opacity, subtitle_bg_padding
        ]

        # 실시간 미리보기: 설정 변경 시 자동 업데이트
        for component in [position_select, use_subtitle_bg, resolution_select]:
            component.change(
                fn=generate_preview,
                inputs=preview_inputs,
                outputs=[preview_image]
            )

        for num_input in [font_size_slider, subtitle_bg_opacity, subtitle_bg_padding, subtitle_offset_x, subtitle_offset_y]:
            num_input.change(
                fn=generate_preview,
                inputs=preview_inputs,
                outputs=[preview_image]
            )

        background_file.change(
            fn=generate_preview,
            inputs=preview_inputs,
            outputs=[preview_image]
        )

        subtitle_text.blur(
            fn=generate_preview,
            inputs=preview_inputs,
            outputs=[preview_image]
        )

        # 파일 업로드 시 텍스트 자동 로드
        tts_file.change(
            fn=load_tts_text,
            inputs=[tts_file],
            outputs=[tts_text]
        )

        subtitle_file.change(
            fn=load_subtitle_text,
            inputs=[subtitle_file],
            outputs=[subtitle_text]
        )

        # 생성 버튼 클릭
        def generate_content(tts_txt, sub_txt, voice, lang, speed, step,
                             bg_file, res, font, pos, offset_x, offset_y,
                             use_bg, bg_opacity, bg_pad, script_file):
            lang_code = get_lang_code(lang)

            # 출력 파일명 결정 (대본 파일명 기반)
            if script_file is not None:
                base_name = os.path.splitext(os.path.basename(script_file.name))[0]
            else:
                base_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            if bg_file is not None:
                # 영상 생성
                video_path, status = create_video(
                    tts_txt, sub_txt, voice, lang_code, speed, step,
                    bg_file.name, res, font, pos, offset_x, offset_y,
                    use_bg, bg_opacity, bg_pad,
                    output_name=base_name
                )
                # 다운로드 파일도 함께 반환
                return None, video_path, status, gr.update(value=video_path, visible=True) if video_path else gr.update(visible=False)
            else:
                # 음성만 생성
                audio_path, status = synthesize_speech(
                    tts_txt, voice, lang_code, speed, step,
                    output_name=base_name
                )
                # 다운로드 파일도 함께 반환
                return audio_path, None, status, gr.update(value=audio_path, visible=True) if audio_path else gr.update(visible=False)

        generate_btn.click(
            fn=generate_content,
            inputs=[
                tts_text, subtitle_text, voice_select, lang_select,
                speed_slider, step_slider, background_file, resolution_select,
                font_size_slider, position_select, subtitle_offset_x, subtitle_offset_y,
                use_subtitle_bg, subtitle_bg_opacity, subtitle_bg_padding, tts_file
            ],
            outputs=[audio_output, video_output, status_output, download_file]
        )

    return demo


# 앱 시작
if __name__ == '__main__':
    print("=" * 50)
    print("Supertonic TTS + Video (Gradio 버전)")
    print("=" * 50)
    print(f"출력 폴더: {OUTPUT_DIR}")
    print("=" * 50)

    # TTS 모델 미리 로드 (CPU 모드)
    init_tts()

    # Gradio 앱 시작
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
