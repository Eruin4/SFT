#!/usr/bin/env python3
import os
import json
import time
import torch
import gc
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 인자 파싱 설정
parser = argparse.ArgumentParser(description='모델 훈련 스크립트')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Math-1.5B-Instruct',
                    help='학습을 시작할 기본 모델 경로 (기본값: Qwen/Qwen2.5-Math-1.5B-Instruct)')
args = parser.parse_args()

# 로깅 설정
disable_progress_bar()  # 진행바 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# 실행 카운터 가져오기
RUN_COUNT = os.environ.get('RUN_COUNT', '0')

# 10의 배수인지 확인
IS_MILESTONE = int(RUN_COUNT) % 10 == 0

# 경로 설정
DATA_DIR = "/home/dshs-wallga/pgh/training_data"
BASE_OUTPUT_DIR = "/home/dshs-wallga/pgh/tuned_model_full"

# 10의 배수 실행인 경우에만 run_X 디렉토리 생성, 그렇지 않으면 temp_model만 사용
if IS_MILESTONE:
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_COUNT}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "merged_model")
else:
    # 10의 배수가 아닐 경우 run_X 디렉토리를 생성하지 않고 temp_model만 사용
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "temp")  # 임시 디렉토리 (checkpoint 저장용)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    FINAL_MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, "temp_model")

# 모델 ID 설정 (명령줄 인자에서 받음)
MODEL_ID = args.model_path

# 시작 정보 출력
print(f"사용할 기본 모델 경로: {MODEL_ID}")
print(f"현재 실행 카운터: {RUN_COUNT}, 저장 경로: {OUTPUT_DIR}")
print(f"전체 모델 저장 여부: {'예' if IS_MILESTONE else '아니오 (temp 모델)'}")

def clear_cuda_memory(device=None):
    """GPU 메모리 정리"""
    print(f"CUDA 메모리 정리 중...")
    if device is not None:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()
    time.sleep(1)
    gc.collect()
    print("CUDA 메모리 정리 완료")

def load_dataset():
    """prepare_dataset.py에서 생성한 훈련 데이터셋 로드"""
    # 현재 실행 카운터의 폴더에서 데이터셋을 찾고, 없으면 기본 경로를 사용
    run_dataset_path = os.path.join(DATA_DIR, f"run_{RUN_COUNT}", "training_dataset.json")
    default_dataset_path = os.path.join(DATA_DIR, "training_dataset.json")
    
    dataset_path = run_dataset_path if os.path.exists(run_dataset_path) else default_dataset_path
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
    
    print(f"훈련 데이터셋 로드 중: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"총 {len(data)}개의 훈련 샘플 로드 완료")
    return data

def init_training_model():
    """훈련용 모델 초기화"""
    print(f"모델 초기화 (GPU 1 사용): {MODEL_ID}")
    
    try:
        # 모델 로드 시도
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        print("모델 및 토크나이저 초기화 완료")
        return model, tokenizer
    
    except Exception as e:
        print(f"모델 로드 오류: {str(e)}")
        print("기본 모델(Qwen/Qwen2.5-Math-1.5B-Instruct)로 대체합니다.")
        
        # 오류 발생 시 기본 모델 사용
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        print("기본 모델 및 토크나이저 초기화 완료")
        return model, tokenizer

def tokenize_function(batch, tokenizer):
    """데이터 토큰화 함수"""
    # 프롬프트와 정답을 결합
    texts = []
    for inp, tgt in zip(batch["input"], batch["target"]):
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": tgt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # 토큰화
    tokenized = tokenizer(
        texts,
        truncation=True, 
        padding="max_length",
        max_length=2048, 
        return_attention_mask=True
    )
    
    # 레이블 생성 - 깊은 복사 사용
    tokenized["labels"] = []
    for i in range(len(tokenized["input_ids"])):
        tokenized["labels"].append(tokenized["input_ids"][i].copy())
    
    # 프롬프트 부분 마스킹 (-100으로 설정)
    for i, (inp, tgt) in enumerate(zip(batch["input"], batch["target"])):
        # 시스템 + 사용자 메시지 길이 계산
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": inp}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # 마스크 길이가 배열 크기를 초과하지 않도록 방지
        mask_len = min(prompt_len, len(tokenized["labels"][i]))
        
        # 프롬프트 부분 마스킹
        tokenized["labels"][i][:mask_len] = [-100] * mask_len
    
    return tokenized

def prepare_lora_config():
    """LoRA 설정 준비"""
    print("LoRA 설정 초기화")
    lora_config = LoraConfig(
        r=16,                     # LoRA의 랭크
        lora_alpha=32,            # LoRA의 스케일링 파라미터
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,        # 드롭아웃 비율
        bias="none",              # 바이어스 학습 여부
        task_type="CAUSAL_LM",    # 태스크 타입
        modules_to_save=["embed_tokens"]  # 임베딩 토큰만 저장
    )
    return lora_config

def train_model(train_data, model, tokenizer):
    """모델 훈련"""
    print("모델 훈련 준비")
    
    # 데이터셋 생성 및 분할
    train_dataset = Dataset.from_dict({
        "input": [item["input"] for item in train_data],
        "target": [item["target"] for item in train_data]
    })
    
    # 토큰화 함수 적용
    def process_batch(batch):
        return tokenize_function(batch, tokenizer)
    
    train_dataset = train_dataset.map(
        process_batch,
        batched=True,
        batch_size=16,
        remove_columns=["input", "target"]
    )
    
    # LoRA 설정 및 적용
    lora_config = prepare_lora_config()
    peft_model = get_peft_model(model, lora_config)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,              # 에포크 수
        per_device_train_batch_size=1,   # 배치 크기 (1에서 4로 증가)
        gradient_accumulation_steps=4,   # 그래디언트 누적 단계 (4에서 2로 감소)
        learning_rate=1e-8,              # 학습률
        save_strategy="epoch",           # 저장 전략
        logging_steps=1,                 # 로깅 단계 (매 스텝마다 로그 출력)
        save_total_limit=2,              # 저장할 체크포인트 수
        bf16=True,                       # 혼합 정밀도 훈련
        remove_unused_columns=False,     # 미사용 열 제거 여부
        logging_strategy="steps",        # 로깅 전략
        disable_tqdm=False,              # tqdm 활성화 (진행 표시줄 표시)
        report_to="none",                # 보고 대상
        log_level="info",                # 로그 레벨
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 훈련 시작
    print(f"모델 훈련 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        trainer.train()
    except Exception as e:
        print(f"훈련 중 오류 발생: {str(e)}")
        raise e
    
    # 모델 저장
    lora_save_dir = os.path.join(OUTPUT_DIR, "lora_model")
    print(f"LoRA 모델 저장 중... 경로: {lora_save_dir}")
    peft_model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    
    # 전체 모델 저장 (합병)
    print("LoRA 모델을 기본 모델과 병합 중...")
    merged_model = peft_model.merge_and_unload()
    
    # 저장 경로 결정 (10의 배수 런에만 전체 저장, 그외에는 temp 모델)
    if IS_MILESTONE:
        merged_save_dir = os.path.join(OUTPUT_DIR, "merged_model")
        print(f"마일스톤 실행(run_{RUN_COUNT}): 최종 합병된 모델 저장 중... 경로: {merged_save_dir}")
    else:
        merged_save_dir = os.path.join(BASE_OUTPUT_DIR, "temp_model")
        # 기존 temp 모델 제거
        if os.path.exists(merged_save_dir):
            import shutil
            shutil.rmtree(merged_save_dir)
        os.makedirs(merged_save_dir, exist_ok=True)
        print(f"일반 실행: 임시 모델로 저장 중... 경로: {merged_save_dir}")
    
    # GPU에서 CPU로 모델 이동 후 저장
    merged_model.to("cpu")
    merged_model.save_pretrained(merged_save_dir)
    tokenizer.save_pretrained(merged_save_dir)
    
    # 병합 완료 후 MODEL_PATH_INFO 출력
    print(f"[MODEL_PATH_INFO] {FINAL_MODEL_PATH}")
    
    # 메타데이터 저장 (모든 실행에 대해)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": MODEL_ID,
        "run_count": RUN_COUNT,
        "output_path": FINAL_MODEL_PATH,
        "is_milestone": IS_MILESTONE
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장 완료: {merged_save_dir if IS_MILESTONE else '임시 모델'}")
    
    return merged_model

def main():
    """메인 함수"""
    start_time = time.time()
    print(f"모델 훈련 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 데이터셋 로드
        train_data = load_dataset()
        
        # 2. 훈련 모델 초기화
        model, tokenizer = init_training_model()
        
        # 3. 모델 훈련 및 저장
        merged_model = train_model(train_data, model, tokenizer)
        
        # 4. 메모리 정리
        del model
        del merged_model
        clear_cuda_memory()
        
        print(f"\n전체 처리 완료: {time.time() - start_time:.2f}초 소요")
        print(f"저장된 모델 경로: {FINAL_MODEL_PATH}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    finally:
        # 메모리 정리
        clear_cuda_memory()

if __name__ == "__main__":
    main() 