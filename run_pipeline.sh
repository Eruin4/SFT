#!/bin/bash

# 로그 파일 설정
LOG_DIR="/home/dshs-wallga/pgh/logs"
mkdir -p $LOG_DIR

# 실행 횟수 카운터 파일
COUNTER_FILE="$LOG_DIR/run_counter.txt"
if [ ! -f "$COUNTER_FILE" ]; then
    echo "0" > "$COUNTER_FILE"
fi

# prepare 스킵 플래그 파일
SKIP_PREPARE_FILE="$LOG_DIR/skip_prepare.flag"

# 기본 경로 설정
BASE_MODEL_DIR="/home/dshs-wallga/pgh/tuned_model_full"
TEMP_MODEL_PATH="$BASE_MODEL_DIR/temp_model"

# GPU 메모리 정리 함수
clean_gpu_memory() {
    echo "GPU 메모리 정리 중..."
    nvidia-smi
    sleep 2
    echo "정리 완료"
}

# 첫 실행 여부 확인
FIRST_RUN=true
if [ -f "$LOG_DIR/model_path.txt" ]; then
    FIRST_RUN=false
    MODEL_PATH=$(cat "$LOG_DIR/model_path.txt")
    
    # 이전 경로가 run_X/merged_model 형식이고 temp_model이 존재하면 temp_model 사용
    if [[ "$MODEL_PATH" == *"/run_"*"/merged_model" ]] && [ -d "$TEMP_MODEL_PATH" ]; then
        if [ $(find "$TEMP_MODEL_PATH" -type f | wc -l) -gt 0 ]; then
            echo "temp_model 폴더가 존재하고 파일이 있어 이를 사용합니다."
            MODEL_PATH="$TEMP_MODEL_PATH"
            echo "$MODEL_PATH" > "$LOG_DIR/model_path.txt"
        fi
    fi
else
    # 첫 실행 시 기본 모델 사용
    MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
    echo $MODEL_PATH > "$LOG_DIR/model_path.txt"
fi

# 스크립트 실행 함수
run_pipeline() {
    COUNTER=$(cat "$COUNTER_FILE")
    TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
    
    echo "===== 파이프라인 실행 #$COUNTER - $TIMESTAMP ====="
    echo "현재 사용 모델: $MODEL_PATH"
    
    # 10의 배수 확인
    IS_MILESTONE=$((COUNTER % 10 == 0))
    if [ "$IS_MILESTONE" -eq 1 ]; then
        echo "마일스톤 실행 (#$COUNTER): 전체 모델 저장됨"
    else
        echo "일반 실행: 임시 모델 사용"
    fi
    
    # prepare 단계 스킵 여부 확인
    SKIP_PREPARE=false
    if [ -f "$SKIP_PREPARE_FILE" ]; then
        SKIP_PREPARE=true
        echo "특수 명령 감지: prepare 단계를 건너뜁니다."
        # 플래그 파일 삭제 (1회성 플래그)
        rm "$SKIP_PREPARE_FILE"
        
        # 가장 최근 prepare 로그 파일 찾기
        LATEST_PREPARE_LOG=$(ls -t "$LOG_DIR"/prepare_*.log 2>/dev/null | head -n 1)
        if [ -n "$LATEST_PREPARE_LOG" ]; then
            echo "최근 prepare 로그: $LATEST_PREPARE_LOG"
            # 현재 prepare 로그를 최신 로그의 심볼릭 링크로 생성
            ln -sf "$LATEST_PREPARE_LOG" "$LOG_DIR/prepare_${TIMESTAMP}.log"
            echo "prepare 단계 건너뜀, 이전 로그를 사용합니다."
            PREPARE_STATUS=0
        else
            echo "이전 prepare 로그를 찾을 수 없습니다. prepare 단계를 실행합니다."
            SKIP_PREPARE=false
        fi
    fi
    
    # 1. 데이터셋 준비 스크립트 실행 (스킵하지 않는 경우)
    if [ "$SKIP_PREPARE" = false ]; then
        echo "데이터셋 준비 스크립트 실행 중..."
        # 실행 카운터를 환경 변수로 전달
        RUN_COUNT=$COUNTER python3 /home/dshs-wallga/pgh/prepare_dataset.py --model_path "$MODEL_PATH" > "$LOG_DIR/prepare_${TIMESTAMP}.log" 2>&1
        PREPARE_STATUS=$?
        
        if [ $PREPARE_STATUS -ne 0 ]; then
            echo "데이터셋 준비 중 오류 발생: 코드 $PREPARE_STATUS"
            return 1
        fi
        
        echo "데이터셋 준비 완료, 메모리 정리 중..."
        clean_gpu_memory
        sleep 10  # GPU 메모리 정리를 위한 대기 시간
    fi
    
    # 2. 모델 훈련 스크립트 실행 (최신 모델 경로 전달)
    echo "모델 훈련 스크립트 실행 중..."
    # 실행 카운터를 환경 변수로 전달
    RUN_COUNT=$COUNTER python3 /home/dshs-wallga/pgh/train_model.py --model_path "$MODEL_PATH" > "$LOG_DIR/train_${TIMESTAMP}.log" 2>&1
    TRAIN_STATUS=$?
    
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "모델 훈련 중 오류 발생: 코드 $TRAIN_STATUS"
        # 훈련 실패 시 다음 실행에서 prepare를 건너뛰도록 플래그 생성
        echo "다음 실행에서 prepare 단계를 건너뛰도록 플래그 설정"
        touch "$SKIP_PREPARE_FILE"
        return 1
    fi
    
    echo "모델 훈련 완료, 모델 경로 추출 중..."
    
    # 로그에서 모델 경로 추출
    NEW_MODEL_PATH=$(grep "\[MODEL_PATH_INFO\]" "$LOG_DIR/train_${TIMESTAMP}.log" | awk '{print $2}')
    
    if [ -z "$NEW_MODEL_PATH" ]; then
        echo "경고: 모델 경로를 찾을 수 없습니다. 이전 경로 유지: $MODEL_PATH"
    else
        echo "새 모델 경로 발견: $NEW_MODEL_PATH"
        
        # 모델 폴더가 실제로 존재하는지 확인
        if [ -d "$NEW_MODEL_PATH" ]; then
        MODEL_PATH=$NEW_MODEL_PATH
            echo "$MODEL_PATH" > "$LOG_DIR/model_path.txt"
            echo "모델 경로가 업데이트되었습니다: $MODEL_PATH"
        else
            echo "경고: 모델 경로($NEW_MODEL_PATH)가 존재하지 않습니다. 이전 경로 유지: $MODEL_PATH"
        fi
    fi
    
    echo "모델 훈련 완료, 메모리 정리 중..."
    clean_gpu_memory
    
    # 카운터 증가 및 저장
    COUNTER=$((COUNTER + 1))
    echo $COUNTER > "$COUNTER_FILE"
    
    echo "===== 파이프라인 실행 완료 #$COUNTER ====="
    return 0
}

# 명령줄 인자 처리
if [ "$1" = "--skip-prepare" ]; then
    echo "특수 명령: 다음 실행에서 prepare 단계를 건너뜁니다."
    touch "$SKIP_PREPARE_FILE"
    shift
fi

# 메인 루프 - 자동 재실행
echo "자동 파이프라인 실행 시작 - Ctrl+C로 중단 가능"
echo "초기 모델 경로: $MODEL_PATH"

while true; do
    START_TIME=$(date +%s)
    
    # 파이프라인 실행
    run_pipeline
    STATUS=$?
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [ $STATUS -eq 0 ]; then
        echo "파이프라인 성공적으로 완료 (소요 시간: ${ELAPSED}초)"
        echo "현재 모델 경로: $MODEL_PATH"
        
        echo "다음 실행 준비 중..."
        sleep 30  # 다음 파이프라인 실행 전 대기 시간
    else
        echo "파이프라인 실행 중 오류 발생, 1분 후 재시도..."
        sleep 60  # 오류 발생 시 더 길게 대기
    fi
done 