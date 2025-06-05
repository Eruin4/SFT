#!/usr/bin/env python3
import os
import json
import re
import time
import torch
import gc
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets.utils.logging import disable_progress_bar
import warnings

# CUDA 장치 설정 - GPU 1만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 1 GPU visible

# 인자 파싱 설정
parser = argparse.ArgumentParser(description='데이터셋 준비 스크립트')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Math-1.5B-Instruct',
                    help='사용할 메인 모델 경로 (기본값: Qwen/Qwen2.5-Math-1.5B-Instruct)')
args = parser.parse_args()

# 로깅 설정
disable_progress_bar()  # 진행바 비활성화
warnings.filterwarnings("ignore", category=UserWarning)

# 실행 카운터 가져오기
RUN_COUNT = os.environ.get('RUN_COUNT', '0')

# 결과 저장 경로
BASE_OUTPUT_DIR = "/home/dshs-wallga/pgh/training_data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_COUNT}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 파일 경로
BENCHMARK_PATH = "/home/dshs-wallga/pgh/AIME_benchmark.json"
TRAIN_PATH = "/home/dshs-wallga/pgh/AIME_train.json"

# 모델 ID 설정 (명령줄 인자에서 받음)
MODEL_ID = args.model_path
SMALL_MODEL_PATH = "/home/dshs-wallga/pgh/qwen_finetuned/omni_2_merged"

# 시작 정보 출력
print(f"사용할 모델 경로: {MODEL_ID}")
print(f"Small 모델 경로: {SMALL_MODEL_PATH}")
print(f"현재 실행 카운터: {RUN_COUNT}, 저장 경로: {OUTPUT_DIR}")

def clear_cuda_memory(device_id=None):
    """GPU 메모리 정리"""
    print(f"CUDA 메모리 정리 중...")
    try:
        if device_id is not None:
            # PyTorch는 device_id를 정수로 받음 (예: 0, 1)
            target_device = torch.device(f"cuda:{int(device_id)}")
            print(f"CUDA 장치 {target_device}의 메모리 정리 시도 중...")
            with torch.cuda.device(target_device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize() # 동기화 추가
        else:
            # device_id가 None이면 현재 설정된 CUDA 장치에 대해 수행
            print("현재 (기본) CUDA 장치의 메모리를 정리합니다.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize() # 동기화 추가
    except Exception as e:
        # 특정 장치 컨텍스트 설정/사용 중 오류 발생 시 폴백
        print(f"메모리 정리 중 오류 발생 (장치 ID: {device_id}): {e}. 사용 가능한 경우 현재 장치에서 계속합니다.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() # 동기화 추가

    gc.collect() # 파이썬 가비지 컬렉션
    time.sleep(1) # 시스템이 안정화될 시간을 줌
    print("CUDA 메모리 정리 완료")

def init_models():
    """모델 초기화"""
    clear_cuda_memory(device_id=0)  # CUDA_VISIBLE_DEVICES가 설정되어 있으므로 cuda:0이 물리적 GPU 1을 가리킴
    print("vLLM 모델 초기화 (GPU 사용)")
    vllm_model = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=3072,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        device="cuda:0"  # CUDA_VISIBLE_DEVICES가 설정되어 있으므로 cuda:0이 물리적 GPU 1을 가리킴
    )
    
    print("Small 모델 초기화 (GPU 사용)")
    small_model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map='auto',  # 'auto'로 설정하여 최적 장치 선택
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    small_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_PATH)
    small_tokenizer.pad_token_id = small_tokenizer.eos_token_id
    small_tokenizer.padding_side = "left"
    
    return vllm_model, tokenizer, small_model, small_tokenizer

def process_batch(vllm_model, tokenizer, small_model, small_tokenizer, prompts, batch_size=16):
    """배치 단위로 문제 처리"""
    print(f"전체 {len(prompts)}개 프롬프트 처리 중...")
    
    # 프롬프트 포맷팅
    batch_texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_texts.append(text)
    
    # vLLM 설정
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|endoftext|>", tokenizer.eos_token],
    )
    
    # 배치 처리 시작
    start_time = time.time()
    outputs = vllm_model.generate(batch_texts, sampling_params)
    
    solutions = []
    extracted_answers = []
    
    # 결과 처리
    for output in outputs:
        model_solution = output.outputs[0].text
        solutions.append(model_solution)
        
        # Small 모델로 답 추출
        inputs_small = small_tokenizer((
            "Extract the correct answer from the solution. A single integer.\n"
            "solution:\n"
            + model_solution[-300:]
            + "\nfinal answer: "
        ), return_tensors='pt').to(small_model.device)
        
        generated_small_ids = small_model.generate(
            **inputs_small, 
            max_new_tokens=10,
            pad_token_id=small_model.config.eos_token_id
        )
        
        small_model_answer = small_tokenizer.decode(generated_small_ids[0], skip_special_tokens=True)
        
        # 정답 추출
        if "final answer: " in small_model_answer:
            answer_part = str(small_model_answer.split("final answer: ")[-1].strip())
        else:
            answer_part = small_model_answer
        
        try:
            match = re.search(r'\d+', answer_part)
            if match:
                extracted_answer = int(match.group())
            else:
                extracted_answer = 0
        except ValueError:
            print("정답 변환 실패")
            extracted_answer = 0
        
        extracted_answers.append(extracted_answer)
    
    elapsed_time = time.time() - start_time
    print(f"배치 처리 완료: {elapsed_time:.2f}초, 평균 {elapsed_time/len(prompts):.2f}초/문제")
    
    return solutions, extracted_answers

def process_benchmarks(vllm_model, tokenizer, small_model, small_tokenizer):
    """AIME 벤치마크 데이터셋 처리"""
    print("\n=== 벤치마크 데이터 처리 시작 ===")
    
    # 데이터 로드
    with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    # 유효한 문제만 필터링
    problems = []
    answers = []
    indices = []
    
    for idx, item in enumerate(benchmark_data):
        try:
            problem = item["prompt"]
            answer = int(item["answer"])
            problems.append(problem)
            answers.append(answer)
            indices.append(idx)
        except (ValueError, KeyError):
            print(f"벤치마크 데이터 #{idx} 처리 불가")
    
    print(f"총 {len(problems)}/{len(benchmark_data)} 개의 유효한 벤치마크 문제")
    
    # 배치 처리
    solutions, extracted_answers = process_batch(
        vllm_model, tokenizer, small_model, small_tokenizer, problems
    )
    
    # 결과 분석
    correct = 0
    results = []
    
    for idx, problem, solution, extracted, answer in zip(indices, problems, solutions, extracted_answers, answers):
        is_correct = extracted == answer
        if is_correct:
            correct += 1
        
        results.append({
            "idx": idx,
            "problem": problem,
            "solution": solution,
            "extracted_answer": extracted,
            "correct_answer": answer,
            "is_correct": is_correct
        })
    
    accuracy = correct / len(problems) if problems else 0
    print(f"벤치마크 정확도: {correct}/{len(problems)} ({accuracy:.2%})")
    
    # 결과 저장 (개별 폴더)
    benchmark_output = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }
    
    benchmark_output_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(benchmark_output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_output, f, ensure_ascii=False, indent=2)
    
    print(f"벤치마크 결과 저장 완료: {benchmark_output_path}")
    return benchmark_output

def create_training_dataset(vllm_model, tokenizer, small_model, small_tokenizer):
    """AIME 훈련 데이터셋 처리 및 학습 데이터 생성"""
    print("\n=== 훈련 데이터 처리 시작 ===")
    
    # 데이터 로드
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 유효한 문제만 필터링
    valid_problems = []
    valid_answers = []
    problem_indices = []
    
    for idx, item in enumerate(train_data):
        try:
            problem = item["prompt"]
            answer = int(item["answer"])
            valid_problems.append(problem)
            valid_answers.append(answer)
            problem_indices.append(idx)
        except (ValueError, KeyError):
            continue
    
    print(f"총 {len(valid_problems)}/{len(train_data)} 개의 유효한 훈련 문제")
    
    # 전체 데이터를 한 번에 처리
    print(f"\n모든 {len(valid_problems)}개 문제를 한 번에 처리합니다...")
    
    # 배치 처리
    solutions, extracted_answers = process_batch(
        vllm_model, tokenizer, small_model, small_tokenizer, valid_problems
    )
    
    # 학습 데이터 누적 저장
    collected_data = []
    
    # 결과 처리
    correct_count = 0
    retry_problems = []
    retry_original_problems = []
    retry_answers = []
    
    # 정답 및 오답 처리
    for i, (problem, solution, extracted, answer) in enumerate(
        zip(valid_problems, solutions, extracted_answers, valid_answers)
    ):
        if extracted == answer:
            collected_data.append({
                "input": problem,
                "target": solution
            })
            correct_count += 1
        else:
            # 오답인 경우 재시도 준비
            retry_prompt = f"""I tried to solve this problem:
            {problem}

            My attempt was:
            {solution}

            However, this was incorrect. I know the correct final answer is {answer}.

            Could you please show me the correct step-by-step derivation to reach the answer {answer}?"""
            
            retry_problems.append(retry_prompt)
            retry_original_problems.append(problem)
            retry_answers.append(answer)
    
    print(f"첫 시도 결과: {correct_count}/{len(valid_problems)} 정답")
    
    # 재시도 배치 처리
    if retry_problems:
        print(f"\n재시도: {len(retry_problems)}개 문제")
        retry_solutions, retry_extracted = process_batch(
            vllm_model, tokenizer, small_model, small_tokenizer, retry_problems
        )
        
        # 재시도 결과 처리
        retry_success = 0
        for orig_problem, retry_solution, extracted, expected in zip(
            retry_original_problems, retry_solutions, retry_extracted, retry_answers
        ):
            if extracted == expected:
                collected_data.append({
                    "input": orig_problem,
                    "target": retry_solution
                })
                retry_success += 1
        
        print(f"재시도 결과: {retry_success}/{len(retry_problems)} 성공")
    
    # 최종 데이터셋 저장
    print(f"\n총 {len(collected_data)}개의 학습 데이터 수집 완료")
    
    # 데이터셋 저장 (개별 폴더 + 통합 파일)
    dataset_output_path = os.path.join(OUTPUT_DIR, "training_dataset.json")
    with open(dataset_output_path, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    
    # 전체 결과를 위한 통합 파일 생성
    combined_dataset_path = os.path.join(BASE_OUTPUT_DIR, "training_dataset.json")
    with open(combined_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    
    print(f"학습 데이터셋 저장 완료:")
    print(f"  - 카운터별 경로: {dataset_output_path}")
    print(f"  - 통합 경로: {combined_dataset_path}")
    
    return collected_data

def main():
    """메인 함수"""
    # 시작 시간 기록
    start_time = time.time()
    print(f"데이터셋 준비 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 모델 초기화
    vllm_model, tokenizer, small_model, small_tokenizer = init_models()
    
    try:
        # 1. AIME 벤치마크 처리
        benchmark_results = process_benchmarks(vllm_model, tokenizer, small_model, small_tokenizer)
        
        # 2-4. AIME 훈련 데이터 처리 및 학습 데이터셋 생성
        training_data = create_training_dataset(vllm_model, tokenizer, small_model, small_tokenizer)
        
        # 메타데이터 저장
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": MODEL_ID,
            "benchmark_accuracy": benchmark_results["accuracy"],
            "training_samples": len(training_data),
            "run_count": RUN_COUNT,
            "elapsed_time": time.time() - start_time
        }
        
        metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n전체 처리 완료: {time.time() - start_time:.2f}초 소요")
        print(f"데이터셋 경로: {OUTPUT_DIR}")
        
    finally:
        # 모델 정리
        try:
            del vllm_model
            del small_model
            del tokenizer
            del small_tokenizer
            clear_cuda_memory(device_id=0)  # CUDA_VISIBLE_DEVICES가 설정되어 있으므로 cuda:0이 물리적 GPU 1을 가리킴
        except Exception as e:
            print(f"모델 정리 중 오류: {e}")
            pass

if __name__ == "__main__":
    main() 